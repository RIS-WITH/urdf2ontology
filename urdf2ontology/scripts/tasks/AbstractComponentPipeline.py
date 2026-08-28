from tasks.ComponentLinkIdentificationTask import ComponentLinkIdentificationTask
from tasks.ComponentLinkMountingPointTask import ComponentLinkMountingPointTask 

from tasks.ComponentLinkSingleClassificationTask import ComponentLinkSingleClassificationTask
from tasks.ComponentLinkGroupClassificationTask import ComponentLinkGroupClassificationTask

from tasks.ComponentLinkGroupingTask import ComponentLinkGroupingTask
from tasks.ComponentChainGroupingTask import ComponentChainGroupingTask

from tasks.ComponentLinkSegmentIdentificationTask import SegmentIdentificationTask

from tasks.ComponentNamingTask import ComponentLinkNamingTask


from urdf_tree import *
from AbstractedRobotLinks import *
from LLMInterface import *

from typing import Optional

import numpy as np

CLASSIF_RUNS = 1
OTHER_RUNS = 3
NAMING_RUNS = 1

class AbstractComponentPipeline():
    def __init__(self, keyword, nb_runs, ontology_handler, paths):
        self.keyword = keyword
        self.nb_runs = nb_runs
        self.ontology_handler = ontology_handler
        self.llm_module = LLMInterface("gpt-oss:20b", 0.7) #"gpt-oss:20b", "qwen3:14b" (reasoning), "llama3.1:8b-instruct-q8_0" mistral-nemo:12b (non-reasoning)
        self.paths = paths
   
    # top-down decomposition (Chain Track)
    def run_chain_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, full_verbose: bool = False, save_intermediate: bool = False, load_intermediate: bool = False) :
        # Used for bodyparts abstraction
        robot_leaf_links = urdf_tree.get_leaf_links()
                
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}     
        
        user_input = {f"link_{i}": link for i, link in enumerate(robot_leaf_links_dict["links"])}
        
        if(load_intermediate):
            path = self.paths["eval"]
            robot_components.components = np.load(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
        else:
            print(f"<==== 1. IDENTIFICATION TASK ====>")
            print("Initial Link Input:", robot_leaf_links_dict.get('links'))
            
            component_identif = ComponentLinkIdentificationTask(OTHER_RUNS, self.keyword)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=10, verbose=True)
            print(f"Final Link identification task: {result_identif}")
            
            robot_components.create_component_links(result_identif, "chain")

            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>") # requires several runs (minimum 3) -> batch
            component_mounting = ComponentLinkMountingPointTask(OTHER_RUNS, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                                batch_size = 5, verbose=full_verbose, verbose_thinking = False)
            print(f"Final Link attachment root detection task: {result_mounting}") 
            robot_components.add_attachment_root_components(result_mounting)
                        
            print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
            pre_classification_task = ComponentLinkSingleClassificationTask(CLASSIF_RUNS, self.keyword, "links")
            upper_concept = "BodyPart"
            init_chain_states = {name: upper_concept for name in robot_components.components.keys()}
            chain_states = self.run_iterative_classification(pre_classification_task, init_chain_states, robot_components.components, 3, verbose = full_verbose)
            print(f"Final Pre-Classification Task: {chain_states}")
            
            robot_components.add_preclassification_components(chain_states)
            
            print(f"<==== 4. SEGMENTATION TASK ====>")
            segmentation_task = SegmentIdentificationTask(OTHER_RUNS+2, self.keyword, "kinematic_csv")
            segments_output = self.llm_module.run_task(segmentation_task, robot_components.components, verbose = True, verbose_thinking = False)
            print(segments_output)
            
            segments = robot_components.create_segments_chain_canonical(segments_output)
            
            flat_segments = {seg.id: seg for chain in robot_components.components.values() for seg in chain.segments}
            upper_concept = "BodyPartSubpart"
            init_segment_states = {seg.id: upper_concept for seg in flat_segments.values()}
            
            print(f"<==== 5. POST-CLASSIFICATION SEGMENTS TASK ====>")
            post_classification_task = ComponentLinkSingleClassificationTask(CLASSIF_RUNS, self.keyword, "links")
            #upper_concept = post_classification_task.parent_context + "BodyPart"        
            segment_states = self.run_iterative_classification(post_classification_task, init_segment_states, flat_segments, 3, verbose = full_verbose)
            print(f"Final Post-Classification Task: {segment_states}")
            
            robot_components.add_postclassification_segments(segment_states)
            
            print(f"<==== 6. CHAIN GROUPING TASK ====>")
            component_grouping = ComponentChainGroupingTask(OTHER_RUNS+2, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=True, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            
            values = self.process_grouped_chains(result_grouping, robot_components.components, robot_components.merged_components, keyword="body_part")
            
            print(f"<==== 6. LINK NAMING TASK ====>")
            chain_link_naming_task = ComponentLinkNamingTask(self.nb_runs, self.keyword, "chain_component", "links")
            result_chain_naming = self.llm_module.run_task(chain_link_naming_task, robot_components.merged_components, verbose=full_verbose, verbose_thinking=False)
            print(f"Final Chain Link Naming task: {result_chain_naming}")
            robot_components.assign_semantic_names(result_chain_naming)
            
            print("FINAL OUTPUT:")
            for chain in robot_components.merged_components.values():
                print(chain)
                
            # =============== INTERMEDIARY SAVE CHECKPOINT ============= 
            if(save_intermediate):
                np.save(self.paths["eval"] + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.components)
                robot_components.components =  np.load(self.paths["eval"] + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
                
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
            
        orphan_links = self.populate_ontology(urdf_tree, robot_components.merged_components)
        urdf_tree.purge_abstracted_components(final_trees=robot_components.merged_components, orphan_links=orphan_links)
                    
        if(not load_intermediate):
            # Save timing
            report_path = self.paths["eval"] + f"/timing_{urdf_tree.robot_name}_{self.keyword}.json"
            self.llm_module.save_metrics(report_path)
        
        return urdf_tree
    
    # bottom-up assembly (Tree Track)
    def run_tree_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, verbose: bool = False, save_intermediate: bool = False, load_intermediate: bool = False) :
        self.init_urdf = urdf_tree

        robot_leaf_links = urdf_tree.get_leaf_links()
        
        if(len(robot_leaf_links) <= 1):
            return
        
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}
        
        user_input = {f"link_{i}": link for i, link in enumerate(robot_leaf_links_dict["links"])}
               
        if(load_intermediate):
            path = self.paths["eval"]
            robot_components.merged_components = np.load(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
        else:
            print(f"<==== 1. IDENTIFICATION TASK ====>")
            component_identif = ComponentLinkIdentificationTask(OTHER_RUNS+2, self.keyword)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=20, verbose=True)
            print(f"Final Link identification task: {result_identif}")
            robot_components.create_component_links(result_identif, "branch")
            
            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>")
            component_mounting = ComponentLinkMountingPointTask(OTHER_RUNS+2, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                               batch_size = 5, verbose=verbose, verbose_thinking = False)
            print(f"Final Link attachment root detection task: {result_mounting}") 
            robot_components.add_attachment_root_components(result_mounting)
           
            # print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
            # pre_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword, "links")
            # upper_concept = pre_classification_task.parent_context + "Subpart"
            # init_link_states = {name: upper_concept for name in robot_components.components.keys()}
            # link_states = self.run_iterative_classification(pre_classification_task, init_link_states, robot_components.components, 
            #                                                 max_iterations = 3, verbose = verbose)
            # print(f"Final Pre-Classification Task: {link_states}")
            # robot_components.add_preclassification_components(link_states)
            
            print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
            # Instantiate task once
            classification_task = ComponentLinkSingleClassificationTask(CLASSIF_RUNS, self.keyword, "links")

            upper_concept = classification_task.parent_context + "Subpart"
            init_link_states = {name: upper_concept for name in robot_components.components.keys()}
                        
            # Run batched task over input components dictionary
            link_states = self.run_batched_iterative_classification(task=classification_task, 
                                                                    initial_states=init_link_states,
                                                                    target_data_map=robot_components.components,
                                                                    max_iterations=5,
                                                                    sub_batch_size=10,
                                                                    verbose=True)
            print(link_states)
            robot_components.add_preclassification_components(link_states)
            
            print(f"<==== 4. GROUPING TASK ====>")
            component_grouping = ComponentLinkGroupingTask(OTHER_RUNS+2, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=verbose, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            robot_components.create_grouped_components(result_grouping, self.keyword)
            
            print(f"<==== 5. POST-CLASSIFICATION TASK ====>")
            post_classification_task = ComponentLinkGroupClassificationTask(CLASSIF_RUNS, self.keyword, "links")
            upper_concept = post_classification_task.parent_context
            init_group_states = {gid: upper_concept for gid in robot_components.merged_components.keys()}
            group_states = self.run_iterative_classification(post_classification_task, init_group_states, 
                                                             robot_components.merged_components, max_iterations = 3, verbose = verbose)
            print(f"Final Post-Classification Task: {group_states}")
            robot_components.add_postclassification(group_states)
        
            print(f"<==== 6. LINK NAMING TASK ====>")
            component_link_naming = ComponentLinkNamingTask(self.nb_runs, self.keyword, "tree_component", "links")
            result_link_naming = self.llm_module.run_task(component_link_naming, robot_components.merged_components, verbose=True, verbose_thinking=False)
            print(f"Final Link Naming task: {result_link_naming}")
            robot_components.assign_semantic_names(result_link_naming)
            
            print("FINAL OUTPUT:")
            for tree in robot_components.merged_components.values():
                print(tree)
        
            if(save_intermediate):
                path = self.paths["eval"]
                np.save(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.merged_components)

        # ============== Ontology Population & verification ============
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
    
        orphan_links = self.populate_ontology(urdf_tree, robot_components.merged_components)
        urdf_tree.purge_abstracted_components(final_trees=robot_components.merged_components, orphan_links=orphan_links)
       
        
        if(not load_intermediate):
            report_path = self.paths["eval"] + f"/timing_{urdf_tree.robot_name}_{self.keyword}.json"
            self.llm_module.save_metrics(report_path)
        
        return urdf_tree
    
    # classification
    def run_iterative_classification(self, task, initial_states, target_data_map, max_iterations=5, verbose=False):
        """
        Refines component classifications iteratively by walking down the ontology tree hierarchy.
        Works universally for Groups, TreeComponents, and ChainComponents.
        
        :param task: The LLM task configuration block. (SingleLinkClassification or GroupClassification)
        :param initial_states: Dict[item_id, current_ontology_concept]
        :param target_data_map: Dict[item_id, RobotComponent/ChainComponent/TreeComponent]
        """
        states = initial_states.copy()
        finished_items = set()
        
        for iteration in range(max_iterations):
            batches = {}
            
            # 1. Batch items based on their current depth in the ontology tree
            for item_id, current_concept in states.items():
                if item_id in finished_items:
                    continue
                    
                # Query the ontology handler for the immediate next layer of subclasses
                sub_concepts = self.ontology_handler.classes.getDown(current_concept, 1)
                
                # If there are no deeper specializations, we've hit a leaf node
                if not sub_concepts or len(sub_concepts) <= 1:
                    finished_items.add(item_id)
                    continue

                if current_concept not in batches:
                    batches[current_concept] = {"items": {}, "sub_types": sub_concepts}
                
                batches[current_concept]["items"][item_id] = target_data_map[item_id]
                
            if not batches:
                break

            changes_found = False

            # 2. Dispatch batches to the LLM module for localized subclass refinement
            for parent_concept, batch_data in batches.items():
                task.parent_context = parent_concept
                task.valid_types = batch_data["sub_types"]
                
                if verbose:
                    print(f"Iteration {iteration}: Refining {list(batch_data['items'].keys())} under '{parent_concept}'")
                
                # Execution call to local model
                results = self.llm_module.run_task(task, batch_data["items"], verbose=verbose)

                # 3. Synchronize State Changes
                for item_id, new_type in results.items():
                    old_type = states[item_id]
                    
                    # Ensure the LLM picked an existing, valid child class from our ontology
                    if new_type in batch_data["sub_types"] and new_type != old_type:
                        states[item_id] = new_type
                        changes_found = True
                        
                        target_data_map[item_id].type = new_type
                    else:
                        # No valid change or same class returned means this item is finalized
                        finished_items.add(item_id)

            if not changes_found:
                break

        return states
    
    def run_batched_iterative_classification(self, task: Any, initial_states: Dict[str, str], target_data_map: Dict[str, Any], 
                                     max_iterations: int = 5,
                                     nb_runs = int,
                                     sub_batch_size: Optional[int] = 5,
                                     verbose: bool = False) -> Dict[str, str]:
        
        """Refines component classifications iteratively by walking down the ontology tree hierarchy, using sub-batching per hierarchy layer to avoid LLM context overflow.

        :param task: The LLM task configuration block.
        :param initial_states: Dict[item_id, current_ontology_concept]
        :param target_data_map: Dict[item_id,
        RobotComponent/ChainComponent/TreeComponent]
        :param max_iterations: Maximum depth/iterations to walk down the
        ontology.
        :param nb_runs: Number of runs
        :param sub_batch_size: Max number of components per prompt payload in
        run_batched_task.
        :param verbose: Print debug output.
        :return: Dict[item_id, final_ontology_concept]
        """
        states = initial_states.copy()
        finished_items: Set[str] = set()

        for iteration in range(max_iterations):
            batches: Dict[str, Dict[str, Any]] = {}

            # 1. Group items based on their current node/parent in the ontology tree
            for item_id, current_concept in states.items():
                if item_id in finished_items:
                    continue

                # Query the ontology handler for the immediate next layer of subclasses
                sub_concepts = self.ontology_handler.classes.getDown(current_concept, 1)

                # Filter out the parent concept itself if returned by getDown
                valid_sub_types = [c for c in sub_concepts if c != current_concept]

                # If there are no deeper specializations, we've hit a leaf node
                if not valid_sub_types:
                    finished_items.add(item_id)
                    continue

                if current_concept not in batches:
                    batches[current_concept] = {
                        "items": {},
                        "sub_types": valid_sub_types,
                    }

                batches[current_concept]["items"][item_id] = target_data_map[item_id]

            # If no items are eligible for further refinement, terminate hierarchy walk
            if not batches:
                break

            changes_found = False

            # 2. Dispatch each hierarchy parent group to the LLM module using sub-batching
            for parent_concept, batch_data in batches.items():
                items_to_classify = batch_data["items"]
                sub_types = batch_data["sub_types"]

                # Configure the task context for the current level of the ontology
                task.parent_context = parent_concept
                task.valid_types = sub_types

                if verbose:
                    print(f"\n--- Iteration {iteration} | Parent Concept: '{parent_concept}' ---")
                    print(f"Target Items ({len(items_to_classify)}): {list(items_to_classify.keys())}")
                    print(f"Allowed Sub-Types: {sub_types}")

                # Call run_batched_task to process items_to_classify in smaller chunks
                if hasattr(self.llm_module, "run_batched_task"):
                    results = self.llm_module.run_batched_task(task=task, 
                                                               user_input=items_to_classify, 
                                                               batch_size=sub_batch_size,
                                                               verbose=verbose)
                else:
                    # Direct fallback if llm_module uses standard task execution
                    results = self.llm_module.run_task(task, items_to_classify, verbose=verbose)

                if not results or not isinstance(results, dict):
                    if verbose:
                        print(f"[WARNING] No valid classification results returned under '{parent_concept}'. Finalizing items.")
                    for item_id in items_to_classify.keys():
                        finished_items.add(item_id)
                    continue

                # 3. Synchronize State Changes & Track Refinement
                for item_id, new_type in results.items():
                    old_type = states.get(item_id)

                    # Ensure the LLM picked an existing, valid child class from our ontology
                    if new_type in sub_types and new_type != old_type:
                        states[item_id] = new_type
                        changes_found = True

                        # Update classification attribute on target object if present
                        if hasattr(target_data_map[item_id], "type"):
                            target_data_map[item_id].type = new_type
                    else:
                        # Item remained at parent level or returned invalid concept -> finalize at parent node
                        finished_items.add(item_id)

            if not changes_found:
                if verbose:
                    print( f"Iteration {iteration}: No further subclass refinements found. Stopping early.")
                break

        return states

    # ontology population
    def populate_chain_assembly(self, chain_component: Any) -> str:
        comp_name = (getattr(chain_component, "semantic_name", None) or getattr(chain_component, "id", None))
        mount_name = getattr(chain_component, "attachment_root", None)
        component_links = list(getattr(chain_component, "all_links", []))

        if not comp_name:
            raise ValueError("ChainComponent must have a semantic_name or id.")

        # 1. Assert Main Component Concept & Inheritance
        self.ontology_handler.feeder.addConcept(comp_name)

        comp_type = getattr(chain_component, "type", None)
        if comp_type:
            self.ontology_handler.feeder.addInheritage(comp_name, comp_type)
            print(f"ASSERTION: {comp_name} -> isA -> {comp_type}")

        if mount_name:
            self.ontology_handler.feeder.addDataRelation(comp_name, "mountsOn", "string", mount_name)
            print(f"ASSERTION: {comp_name} -> mountsOn -> {mount_name}")

        assembly_base = getattr(chain_component, "assembly_base_link", None)
        if assembly_base:
            self.ontology_handler.feeder.addDataRelation(comp_name, "hasAssemblyLinkFrame", "string", assembly_base)
            print(f"ASSERTION: {comp_name} -> hasAssemblyLinkFrame -> {assembly_base}")

        # 2. Extract and Normalize Segments
        segments = getattr(chain_component, "segments", [])
        ordered_segment_items = []

        if segments:
            for seg in segments:
                if isinstance(seg, dict):
                    seg_name = seg.get("semantic_name") or seg.get("id") or seg.get("link_name")
                    seg_type = seg.get("type", "SegmentElement")
                    link_frames = seg.get("link_chain", [])
                elif(isinstance(seg, RobotComponent)):
                    seg_name = (getattr(seg, "semantic_name", None) or getattr(seg, "id", None) or getattr(seg, "link_name", None))
                    seg_type = getattr(seg, "type", "SegmentElement")
                    link_frames = getattr(seg, "link_chain", [])
                
                if seg_name:
                    ordered_segment_items.append((seg_name, seg_type, link_frames))
        else:
            # Fallback: Create explicit Segment individuals from raw links
            for link_name in component_links:
                seg_name = f"{link_name}_segment" if link_name == comp_name else link_name
                ordered_segment_items.append((seg_name, "SegmentElement", link_name))

        # 3. Assert Segment Individuals & Link Frames
        for seg_name, seg_type, link_frame in ordered_segment_items:
            self.ontology_handler.feeder.addConcept(seg_name)
            if seg_type:
                self.ontology_handler.feeder.addInheritage(seg_name, seg_type)
                print(f"ASSERTION: {seg_name} -> isA -> {seg_type}")
            
            for link_frame in link_frames:
                self.ontology_handler.feeder.addDataRelation(seg_name, "hasLinkFrame", "string", link_frame)
                print(f"ASSERTION: {seg_name} -> hasLinkFrame -> {link_frame}")

        # 4. Connect Chain Relations based on Segment Count
        num_segments = len(ordered_segment_items)

        # CASE A: Single Segment -> Simple direct relation
        if num_segments == 1:
            single_seg_name = ordered_segment_items[0][0]
            self.ontology_handler.feeder.addObjectRelation(comp_name, "hasSegment", single_seg_name)
            print(f"ASSERTION: {comp_name} -> hasSegment -> {single_seg_name}")

        # CASE B: Multiple Segments -> Full chain positional tracking
        elif num_segments > 1:
            first_seg_name = ordered_segment_items[0][0]
            last_seg_name = ordered_segment_items[-1][0]

            self.ontology_handler.feeder.addObjectRelation(comp_name, "hasInitialSegment", first_seg_name)
            print(f"ASSERTION: {comp_name} -> hasInitialSegment -> {first_seg_name}")
            self.ontology_handler.feeder.addObjectRelation(comp_name, "hasTerminalSegment", last_seg_name)
            print(f"ASSERTION: {comp_name} -> hasInitialSegment -> {last_seg_name}")

            for i in range(num_segments - 1):
                curr_seg_name = ordered_segment_items[i][0]
                next_seg_name = ordered_segment_items[i + 1][0]
                self.ontology_handler.feeder.addObjectRelation(curr_seg_name, "hasNextSegment", next_seg_name)
                print(f"ASSERTION: {curr_seg_name} -> hasNextSegment -> {next_seg_name}")

        return comp_name

    def populate_tree_branch(self, sub_component: RobotComponent, tree_component: TreeComponent, assembly_name: str):
        # If a branch is a ChainComponent (for clustered chains)
        if isinstance(sub_component, ChainComponent) or hasattr(sub_component, "segments"):
            branch_name = self.populate_chain_assembly(sub_component)
            self.ontology_handler.feeder.addObjectRelation(assembly_name, "hasSubPart", branch_name)
            print("ASSERTION:", assembly_name, " -> hasSubPart ->", branch_name)
        else:
            sub_comp_name = getattr(sub_component, "semantic_name", None) or getattr(sub_component, "id", "sub_part")
        
            # If skipping full subpart instantiation for single-branch components, 
            # make sure to map the link frame to the parent component!
            links = getattr(sub_component, "link_chain", []) or getattr(sub_component, "links", [])
            
            if len(tree_component.branches) == 1 and len(links) == 1:
                # Assert link frame directly to main component rather than skipping silently
                self.ontology_handler.feeder.addDataRelation(assembly_name, "hasLinkFrame", "string", links[0])
                print(f"ASSERTION: {assembly_name} -> hasLinkFrame -> {links[0]}")
            else:
                self.ontology_handler.feeder.addConcept(sub_comp_name)
                self.ontology_handler.feeder.addInheritage(sub_comp_name, getattr(sub_component, "type", "Subpart"))
                print("ASSERTION:", sub_comp_name, " -> isA ->", getattr(sub_component, "type", "Subpart"))
                
                self.ontology_handler.feeder.addObjectRelation(assembly_name, "hasSubPart", sub_comp_name)
                print("ASSERTION:", assembly_name, "-> hasSubPart ->", sub_comp_name)
                
                for link in links:
                    self.ontology_handler.feeder.addDataRelation(sub_comp_name, "hasLinkFrame", "string", link)
                    print("ASSERTION:", sub_comp_name, " -> hasLinkFrame ->", link)
        
    def populate_tree_assembly(self, tree_component : TreeComponent) -> str:
        comp_name = getattr(tree_component, "semantic_name", None)
        mount_name = getattr(tree_component, "attachment_root", None)
        
        if comp_name:
            self.ontology_handler.feeder.addConcept(comp_name)

        if getattr(tree_component, "type", None):
            self.ontology_handler.feeder.addInheritage(comp_name, tree_component.type)
            print("ASSERTION:", comp_name, " -> isA -> ", tree_component.type)

        if mount_name:
            self.ontology_handler.feeder.addDataRelation(comp_name, "mountsOn", "string", mount_name)
            print("ASSERTION:", comp_name, " -> mountsOn ->", mount_name)

        if getattr(tree_component, "assembly_base_link", None):
            self.ontology_handler.feeder.addDataRelation(comp_name, "hasAssemblyLinkFrame", "string", tree_component.assembly_base_link)
            print("ASSERTION:", comp_name, " -> hasAssemblyLinkFrame ->", tree_component.assembly_base_link)
        
        branches = getattr(tree_component, "branches", [])
        for sub_comp in branches:
            self.populate_tree_branch(sub_comp, tree_component, comp_name)
            
            
        return comp_name

    def populate_ontology(self, urdf_tree : URDFTree, abstracted_components: Dict[str, RobotComponent]) -> Set[str]:

        all_orphan_nodes: Set[str] = set()

        # 1. Build local set of classified links for this stage
        all_classified_links: Set[str] = set()
        for comp in abstracted_components.values():
            all_classified_links.update(getattr(comp, "all_links", []))
        
        # populate the components 
        for comp_id, component in abstracted_components.items():
            component_links = list(getattr(component, "all_links", []))
            if not component_links:
                    continue
                
            if isinstance(component, ChainComponent) or hasattr(component, "segments"):
                comp_name = self.populate_chain_assembly(component)
                
            elif isinstance(component, TreeComponent) or hasattr(component, "branches"):
                comp_name = self.populate_tree_assembly(component)
                
            # 2. Discover Orphans Attached to Tree Links
            for link_name in component_links:
                if link_name not in urdf_tree.links:
                    continue

                comp_link_obj = urdf_tree.links[link_name]
                external_children = [child for child in comp_link_obj.children if child.name not in all_classified_links]

                for orphan_entry_link in external_children:
                    self._process_and_register_orphan_subtree(
                        current_link=orphan_entry_link,
                        parent_comp_name=comp_name,
                        all_classified_links=all_classified_links,
                        orphan_set=all_orphan_nodes,
                    )
                
        return all_orphan_nodes
    
    # verification
    def _process_and_register_orphan_subtree(
        self,
        current_link: Any,
        parent_comp_name: str,
        all_classified_links: Set[str],
        orphan_set: Set[str],
    ) -> None:
        """
        Recursively traverses unclassified subtrees, registers orphan link concepts,
        and connects them to their parent component in the ontology feeder.
        """
        link_name = getattr(current_link, "name", str(current_link))

        # Stop traversal if link is already classified elsewhere or already registered as an orphan
        if link_name in all_classified_links or link_name in orphan_set:
            return

        # 1. Register orphan node concept and properties in ontology
        self.ontology_handler.feeder.addConcept(link_name)
        self.ontology_handler.feeder.addInheritage(link_name, "OrphanLinkNode")
        self.ontology_handler.feeder.addObjectRelation(parent_comp_name, "hasOrphanNode", link_name)
        self.ontology_handler.feeder.addDataRelation(link_name, "hasLinkFrame", "string",  link_name)

        # 2. Track processed orphan link
        orphan_set.add(link_name)

        # 3. Recursively process unclassified downstream children
        children = getattr(current_link, "children", [])
        for child in children:
            child_name = getattr(child, "name", str(child))
            if child_name not in all_classified_links:
                self._process_and_register_orphan_subtree(
                    current_link=child,
                    parent_comp_name=parent_comp_name,
                    all_classified_links=all_classified_links,
                    orphan_set=orphan_set,
                )
                
    def get_part_subproperty_from_ontology(self, comp_name: str) -> str:
        """
        Queries the ontology for the class hierarchy of the given individual 
        and returns the corresponding subproperty of 'part'.
        """
        try:
            # Retrieve all direct and indirect superclasses/types for the individual
            instance_classes = self.ontology_handler.individuals.getUp(comp_name)
        except Exception as e:
            print(f"Warning: Failed to fetch class hierarchy for '{comp_name}': {e}")
            instance_classes = []

        # Flatten classes to lower-case string representations for checking
        class_hierarchy = [str(cls).lower() for cls in instance_classes]

        # Check hierarchy concepts against target upper-component types
        if any("sensor" in cls for cls in class_hierarchy):
            return "hasSensor"
        elif any("endeffector" in cls or "effector" in cls for cls in class_hierarchy):
            return "hasEndEffector"
        elif any("locomotion" in cls for cls in class_hierarchy):
            return "hasLocomotion"
        elif any("bodypart" in cls or "body" in cls for cls in class_hierarchy):
            return "hasBodyPart"

        # Default fallback if no specific class matches
        return "part"

    # upper part inter-connected assertions
    def resolve_and_assert_component_hierarchy(self, all_components: Dict[str, Any], robot_individual_name: str) -> None:
        """
        Final Post-Processing Pass:
        Resolves inter-component hierarchy across all extracted steps.
        Asserts:
        - Upper-level components -> (Robot Instance -> part -> Component)
        - Sub-components -> (Parent Component -> hasSubSystem -> Child Component)
        """
        resolved_components = self.resolve_component_hierarchy(all_components)

        robot_instance_name = robot_individual_name + "_robot"
        self.ontology_handler.feeder.addConcept(robot_instance_name)
        print("Creating the main robot individual: ", robot_instance_name)
            
        relations_count = 0
        for comp_id, comp_data in resolved_components.items():
            is_dict = isinstance(comp_data, dict)

            # Get current component semantic name or ID
            comp_name = (
                comp_data.get("semantic_name")
                if is_dict
                else getattr(comp_data, "semantic_name", None)
            ) or comp_id

            # Ensure concept exists in ontology before asserting relations
            self.ontology_handler.feeder.addConcept(comp_name)

            # Retrieve parent component ID resolved by resolve_component_hierarchy
            parent_comp_id = (
                comp_data.get("parent_component_id")
                if is_dict
                else getattr(comp_data, "parent_component_id", None)
            )

            # CASE 1: Child Component -> Has a valid parent component in the assembly
            if parent_comp_id and parent_comp_id in resolved_components and parent_comp_id != comp_id:
                parent_data = resolved_components[parent_comp_id]
                parent_name = (
                    (
                        parent_data.get("semantic_name")
                        if isinstance(parent_data, dict)
                        else getattr(parent_data, "semantic_name", None)
                    )
                    or parent_comp_id
                )

                # Ensure parent concept exists
                self.ontology_handler.feeder.addConcept(parent_name)

                # Assert Parent -> hasSubSystem -> Child
                self.ontology_handler.feeder.addObjectRelation(parent_name, "hasSubSystem", comp_name)
                print(f"ASSERTION: {parent_name} -> hasSubSystem -> {comp_name}")
                relations_count += 1

            # CASE 2: Top-Level / Upper Component -> Directly attached to the main robot chassis
            else:
                relation_pred = self.get_part_subproperty_from_ontology(comp_name)
                self.ontology_handler.feeder.addObjectRelation(robot_instance_name, relation_pred, comp_name)
                print(f"ASSERTION: {robot_instance_name} -> {relation_pred} -> {comp_name}")
                relations_count += 1

        print(
            f"\n[HIERARCHY RESOLVED] Asserted {relations_count} hierarchy relations in ontology."
        )

    def resolve_component_hierarchy(self, all_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds direct component-to-component parent/child relationships based on 
        which component owns the mount_link of another component.
        """
        link_owner_map: Dict[str, str] = {}

        def extract_links_from_obj(obj: Any) -> List[str]:
            """Recursively extracts all raw link names from a component/sub-component."""
            links = []
            if isinstance(obj, dict):
                links.extend(obj.get("all_links", []))
                links.extend(obj.get("link_chain", []))
                for branch in obj.get("branches", []):
                    links.extend(extract_links_from_obj(branch))
                for seg in obj.get("segments", []):
                    links.extend(extract_links_from_obj(seg))
            else:
                links.extend(getattr(obj, "all_links", []))
                links.extend(getattr(obj, "link_chain", []))
                
                # Recursive check for TreeComponent branches
                for branch in getattr(obj, "branches", []):
                    links.extend(extract_links_from_obj(branch))
                # Recursive check for ChainComponent segments
                for seg in getattr(obj, "segments", []):
                    links.extend(extract_links_from_obj(seg))
            return list(set(links))

        # 1. Map every link frame -> owning component ID
        for comp_id, comp_data in all_components.items():
            comp_links = extract_links_from_obj(comp_data)
            for link_name in comp_links:
                if link_name:
                    link_owner_map[link_name] = comp_id

        # 2. Resolve parent component for each entity
        for comp_id, comp_data in all_components.items():
            if isinstance(comp_data, dict):
                mount_link = (
                    comp_data.get("mount_link") 
                    or comp_data.get("attachment_root") 
                    or comp_data.get("assembly_base_link")
                )
            else:
                mount_link = (
                    getattr(comp_data, "mount_link", None) 
                    or getattr(comp_data, "attachment_root", None)
                    or getattr(comp_data, "assembly_base_link", None)
                )

            parent_comp_id = link_owner_map.get(mount_link) if mount_link else None

            # Prevent self-referential parent loops
            if parent_comp_id and parent_comp_id != comp_id:
                if isinstance(comp_data, dict):
                    comp_data["parent_component_id"] = parent_comp_id
                else:
                    setattr(comp_data, "parent_component_id", parent_comp_id)
            else:
                if isinstance(comp_data, dict):
                    comp_data["parent_component_id"] = None
                else:
                    setattr(comp_data, "parent_component_id", None)

        return all_components
    
    def process_grouped_chains(self, aggregated_groups: List[List[str]], components, merged_components, keyword: str = "bodypart") -> Dict[str, RobotComponent]:
        """
        Processes the aggregated output of ComponentChainGroupingTask.
        
        - Multi-chain groups (len > 1) are merged into a new TreeComponent referencing 
        the existing ChainComponents as branches.
        - Single-chain groups (len == 1) remain untouched as standalone ChainComponents.
        """
        macro_components: Dict[str, RobotComponent] = {}

        for i, chain_ids in enumerate(aggregated_groups):
            # Retrieve the actual ChainComponent objects from self.components using IDs
            matched_chains: List[ChainComponent] = [components[cid] for cid in chain_ids if cid in components]

            if not matched_chains:
                continue

            # -----------------------------------------------------------------
            # CASE 1: Single Chain -> Leave as untouched ChainComponent
            # -----------------------------------------------------------------
            if len(matched_chains) == 1:
                standalone_chain = matched_chains[0]
                macro_components[standalone_chain.id] = standalone_chain

            # -----------------------------------------------------------------
            # CASE 2: Merged Chains -> Wrap into a new upper TreeComponent
            # -----------------------------------------------------------------
            else:
                tree_id = f"{keyword}_tree_{i}"
                all_entity_links: Set[str] = set()
                collected_chain_types: Set[str] = set()
                root_attachment = None

                for chain_comp in matched_chains:
                    # Aggregate links from all branches
                    all_entity_links.update(getattr(chain_comp, "all_links", []) or getattr(chain_comp, "link_chain", []))
                    
                    # Collect types for tree classification
                    if getattr(chain_comp, "type", None):
                        collected_chain_types.add(chain_comp.type)
                    
                    # Extract attachment root from inner segments or chain
                    for seg in getattr(chain_comp, "segments", []):
                        if not root_attachment and getattr(seg, "attachment_root", None):
                            root_attachment = seg.attachment_root
                        if hasattr(seg, "parent_chain_types") and seg.parent_chain_types:
                            collected_chain_types.update(seg.parent_chain_types)

                # Determine base link and attachment root from the primary branch
                primary_chain = matched_chains[0]
                base_link = getattr(primary_chain, "assembly_base_link", None)
                if not base_link and getattr(primary_chain, "link_chain", None):
                    base_link = primary_chain.link_chain[0]

                if not root_attachment:
                    root_attachment = getattr(primary_chain, "attachment_root", "")

                tree_type = ( next(iter(collected_chain_types))if collected_chain_types else keyword.capitalize())

                # Create upper TreeComponent referencing the existing ChainComponents
                macro_tree = TreeComponent(id=tree_id, type=tree_type, assembly_base_link=base_link, attachment_root=root_attachment or "",
                    branches=matched_chains, 
                    all_links=all_entity_links
                )

                macro_components[tree_id] = macro_tree

        # Store finalized components in merged_components
        merged_components.update(macro_components)
        
        return macro_components
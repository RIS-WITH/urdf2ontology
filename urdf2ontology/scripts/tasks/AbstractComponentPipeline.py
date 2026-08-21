from tasks.ComponentLinkIdentificationTask import ComponentLinkIdentificationTask
from tasks.ComponentLinkMountingPointTask import ComponentLinkMountingPointTask 
from tasks.ComponentLinkSingleClassificationTask import ComponentLinkSingleClassificationTask
from tasks.ComponentLinkGroupingTask import ComponentLinkGroupingTask
from tasks.ComponentLinkGroupClassificationTask import ComponentLinkGroupClassificationTask
from tasks.SegmentGroupingTask import ChainLinkGroupingTask

from tasks.ComponentLinkSegmentIdentificationTask import SegmentIdentificationTask

# from tasks.ComponentNamingTask import ComponentLinkNamingTask
from tasks.LinkNamingTask import ComponentLinkNamingTask

from urdf_tree import *
from AbstractedRobotLinks import *
from LLMInterface import *

import numpy as np

class AbstractComponentPipeline():
    def __init__(self, keyword, nb_runs, ontology_handler, paths):
        self.keyword = keyword
        self.nb_runs = nb_runs
        self.ontology_handler = ontology_handler
        self.llm_module = LLMInterface("gpt-oss:20b", 0.7) #"gpt-oss:20b", "qwen3:14b" (reasoning), "llama3.1:8b-instruct-q8_0" mistral-nemo:12b (non-reasoning)
        self.paths = paths
   
    # top-down
    def run_chain_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, full_verbose: bool = False, save_intermediate: bool = False, load_intermediate: bool = False) :
        # Used for bodyparts abstraction
        robot_leaf_links = urdf_tree.get_leaf_links()
        
        if(len(robot_leaf_links) <= 1):
            return
                
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}     
        
        user_input = {f"link_{i}": link for i, link in enumerate(robot_leaf_links_dict["links"])}
        
        if(load_intermediate):
            path = self.paths["main_path"]
            robot_components.components = np.load(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
        else:
            print(f"<==== 1. IDENTIFICATION TASK ====>")
            print("Initial Link Input:", robot_leaf_links_dict.get('links'))
            
            component_identif = ComponentLinkIdentificationTask(self.nb_runs*3, self.keyword)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=10, verbose=True)
            print(f"Final Link identification task: {result_identif}")
            
            robot_components.create_component_links(result_identif, "chain")

            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>") # requires several runs (minimum 3) -> batch
            component_mounting = ComponentLinkMountingPointTask(self.nb_runs*3, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                                batch_size = 5, verbose=full_verbose, verbose_thinking = False)
            print(f"Final Link attachment root detection task: {result_mounting}") 
            robot_components.add_attachment_root_components(result_mounting)
                        
            print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
            pre_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword, "links")
            upper_concept = "BodyPart"
            init_chain_states = {name: upper_concept for name in robot_components.components.keys()}
            chain_states = self.run_iterative_classification(pre_classification_task, init_chain_states, robot_components.components, 3, verbose = full_verbose)
            print(f"Final Pre-Classification Task: {chain_states}")
            
            robot_components.add_preclassification_components(chain_states)
            
            print(f"<==== 4. SEGMENTATION TASK ====>")
            segmentation_task = SegmentIdentificationTask(self.nb_runs*3, self.keyword, "kinematic_csv")
            segments_output = self.llm_module.run_task(segmentation_task, robot_components.components, verbose = True, verbose_thinking = False)
            print(segments_output)
            
            #robot_components.create_segments_chain(segments_output)
            
            segments = robot_components.create_segments_chain_canonical(segments_output)
            
            print(robot_components.components)
            
            # for comp_id, component in robot_components.components.items():
            #     if component.segments:
            #         # Extract segment IDs in kinematic order
            #         # full_list = []
            #         # for seg in component.segments:
            #         #     for link in seg.link_chain:
            #         #         full_list.append(link)
            #         segment_ids = [seg.id for seg in component.segments]
                    
            #         # Format into sequential text
            #         sequence_str = " then ".join(segment_ids)
            #         print(f"{comp_id} is composed of {sequence_str}")
                    
            flat_segments = {seg.id: seg for chain in robot_components.components.values() for seg in chain.segments}
            upper_concept = "BodyPartSubpart"
            init_segment_states = {seg.id: upper_concept for seg in flat_segments.values()}
            
            print(f"<==== 5. POST-CLASSIFICATION SEGMENTS TASK ====>")
            post_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword, "links")
            #upper_concept = post_classification_task.parent_context + "BodyPart"        
            segment_states = self.run_iterative_classification(post_classification_task, init_segment_states, flat_segments, 3, verbose = full_verbose)
            print(f"Final Post-Classification Task: {segment_states}")
            
            robot_components.add_postclassification_segments(segment_states)
            
            print(robot_components.components)
                    
            # print(f"<==== 6. SEGMENT GROUPING TASK ====>")
            # component_grouping = ComponentLinkGroupingTask(self.nb_runs, self.keyword, "links")
            # result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=full_verbose, verbose_thinking = False)
            # print(f"Final Link grouping task: {result_grouping}")
            
            print(f"<==== 6. CHAIN GROUPING TASK ====>")
            component_grouping = ChainLinkGroupingTask(self.nb_runs*3, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=True, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            
            value = robot_components.build_macro_entities_dict(result_grouping)
                
            #robot_components.create_grouped_components(result_grouping, self.keyword)

            print(f"<==== 6. LINK NAMING TASK ====>")
            # # first string corresponds to the type of entity to name, the second one to the type of input chain to provide
            component_link_naming = ComponentLinkNamingTask(self.nb_runs, self.keyword, "chain_component", "links")
            result_link_naming = self.llm_module.run_task(component_link_naming, robot_components.components, verbose=full_verbose, verbose_thinking=False)
            print(f"Final Link Naming task: {result_link_naming}")
            
            robot_components.assign_semantic_names(result_link_naming, "chain")
            
            # =============== INTERMEDIARY SAVE CHECKPOINT ============= 
            if(save_intermediate):
                np.save(self.paths["eval"] + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.components)
                robot_components.components =  np.load(self.paths["eval"] + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
                
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
        
        for comp in robot_components.components.values():
            print(comp)
                    
        # 1. Populate Ontology for Chains
        orphan_links = self.populate_chain_ontology_new(urdf_tree=urdf_tree, final_chains=robot_components.components)

        # 2. Purge Chain Links & Orphans from URDF
        urdf_tree.purge_abstracted_components(final_trees=robot_components.components, orphan_links=orphan_links)
        # self.populate_ontology_chain_groups(robot_components.components, False)
        

        # urdf_tree.abstract_components_chain(robot_components.components, False)
        
        # Save timing
        report_path = self.paths["eval"] + f"/timing_{urdf_tree.robot_name}_{self.keyword}.json"
        self.llm_module.save_metrics(report_path)
        
        return urdf_tree
    
    # bottom-up
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
            component_identif = ComponentLinkIdentificationTask(self.nb_runs*3, self.keyword)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=10, verbose=True)
            print(f"Final Link identification task: {result_identif}")
            robot_components.create_component_links(result_identif, "branch")
            
            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>")
            component_mounting = ComponentLinkMountingPointTask(self.nb_runs*3, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                               batch_size = 5, verbose=verbose, verbose_thinking = False)
            print(f"Final Link attachment root detection task: {result_mounting}") 
            robot_components.add_attachment_root_components(result_mounting)
           
            print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
            pre_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword, "links")
            upper_concept = pre_classification_task.parent_context + "Subpart"
            init_link_states = {name: upper_concept for name in robot_components.components.keys()}
            link_states = self.run_iterative_classification(pre_classification_task, init_link_states, robot_components.components, 
                                                            max_iterations = 3, verbose = verbose)
            print(f"Final Pre-Classification Task: {link_states}")
            robot_components.add_preclassification_components(link_states)
        
            print(f"<==== 4. GROUPING TASK ====>")
            component_grouping = ComponentLinkGroupingTask(self.nb_runs*3, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=verbose, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            robot_components.create_grouped_components(result_grouping, self.keyword)
            
            print(f"<==== 5. POST-CLASSIFICATION TASK ====>")
            post_classification_task = ComponentLinkGroupClassificationTask(self.nb_runs, self.keyword, "links")
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
            robot_components.assign_semantic_names(result_link_naming, "tree")
        
        if(save_intermediate):
            path = self.paths["eval"]
            np.save(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.merged_components)

        # ============== Ontology Population & verification ============
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
        
        for comp in robot_components.merged_components.values():
            print(comp)
            
        # 1. Populate Ontology 
        orphan_links = self.populate_ontology_tree_new(urdf_tree=urdf_tree, final_trees=robot_components.merged_components)

        # 2. Purge URDF Tree Structure
        urdf_tree.purge_abstracted_components(final_trees=robot_components.merged_components, orphan_links=orphan_links)
        

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

    # ontology population
    def populate_ontology_chain_groups(self, final_chain_groups: Dict[str, RobotComponent], save_intermediate = False):
        for chain_comp in final_chain_groups.values():
            comp_name = chain_comp.semantic_name
            self.ontology_handler.feeder.addConcept(comp_name) # create the upper chain individual (e.g., pepper_left_arm)
            self.ontology_handler.feeder.addInheritage(comp_name, chain_comp.type) # assert its associated type (e.g., Arm(pepper_left_arm))
            
            for segment_comp in chain_comp.segments: # loop over the segments
                segment_name = segment_comp.semantic_name
                self.ontology_handler.feeder.addConcept(segment_name) # create the lower-level individual (e.g., pepper_left_elbow)
                self.ontology_handler.feeder.addInheritage(segment_name, segment_comp.type) # assert its type (e.g., Elbow(pepper_left_elbow))
                
                self.ontology_handler.feeder.addObjectRelation(comp_name, 'hasSubPart', segment_name) # assert upper subpart relation -> hasSubPart(pepper_left_arm, pepper_left_elbow)
                # Need to make sure that the attachment root corresponds to the name used by the naming task.
                self.ontology_handler.feeder.addObjectRelation(chain_comp.attachment_root, 'connectsTo', segment_name) # connectsTo(pepper_left_elbow, pepper_left_forearm)
                
                for link in segment_comp.link_chain: # assert the link frames as data properties
                    self.ontology_handler.feeder.addDataRelation(segment_name, 'hasLinkFrame', "string", link) # i.e., hasLinkFrame(pepper_right_arm, string#"pepper_right_shoulder_frame")
      
    def populate_ontology_tree_groups(self, final_tree_groups: Dict[str, RobotComponent], save_intermediate = False):
        
         for tree_comp in final_tree_groups.values():
            comp_name = tree_comp.semantic_name
            if(comp_name):
                self.ontology_handler.feeder.addConcept(comp_name)
            if(tree_comp.type):
                self.ontology_handler.feeder.addInheritage(comp_name, tree_comp.type) # assert its associated type (e.g., Arm(pepper_left_arm))
            if(tree_comp.attachment_root):
                self.ontology_handler.feeder.addObjectRelation(comp_name, 'mountsOn', tree_comp.attachment_root) # to be done at the end of the URDF analysis
            
            # add the assembly base link if it exists
            if(tree_comp.assembly_base_link):
                self.ontology_handler.feeder.addDataRelation(comp_name, 'hasAssemblyLinkFrame', "string", tree_comp.assembly_base_link)
                
                
            # the tree component has only one branch which is of length 1
            if(len(tree_comp.branches) == 1 & len(tree_comp.branches[0].link_chain) == 1):
                print("Tree component with individual single branch found", tree_comp)
                print("Skipping further sub-element:", tree_comp.branches[0])
                continue
            
            for sub_comp in tree_comp.branches: # loop over the sub components
                sub_comp_name = sub_comp.semantic_name
                self.ontology_handler.feeder.addConcept(sub_comp_name) # to be replaced with semantic name
                self.ontology_handler.feeder.addInheritage(sub_comp_name, sub_comp.type) # assert its type (e.g., Elbow(pepper_left_elbow))
                
                self.ontology_handler.feeder.addObjectRelation(comp_name, 'hasSubPart', sub_comp_name) # link higher indiv with branch
                #self.ontology_handler.feeder.addObjectRelation(tree_comp.attachment_root, 'connectsTo', sub_comp.id) # link sub
                
                for link in sub_comp.link_chain: # assert the link frames as data properties
                    self.ontology_handler.feeder.addDataRelation(sub_comp_name, 'hasLinkFrame', "string", link)
                    
            if(save_intermediate):
                robot_name = self.init_urdf.urdf_path.split('/')[-1].split('.')[0]
                self.ontology_handler.actions.save(self.paths["main_path"] + "/results/abstraction_" + robot_name +  "_" + self.keyword +".owl") 
     
    # main function for ontology population
    def populate_ontology(self, abstracted_robot: AbstractedRobotLinks, ontology_handler):
        
        # 1. create the core individual (e.g., pepper_robot) + type it if it exists in the ontology
        core_robot_individual = abstracted_robot.robot_name + "_core"
        self.ontology_handler.feeder.addConcept(core_robot_individual)
        robot_subclasses = self.ontology_handler.feeder.classes.getDown("Robot")
        
        for robot_model in robot_subclasses:
            if(abstracted_robot.robot_name.lower() in robot_model.lower()):
                print(f"Robot Model found => name: {abstracted_robot.robot_name} with concept: {robot_model}")
                self.ontology_handler.feeder.addInheritage(core_robot_individual, robot_model)
                
        # 2. create the first link into the tree (e.g., pepper_base)
        self.ontology_handler.feeder.addConcept()
        # 3. create the individuals for segments and chains (e.g., pepper_elbow and pepper_right_arm)
        # 4. create the end-effector/sensor/locomotion tree groups
        
        
        return
    
    # extract sub URDFTree              
    def extract_component_as_tree(self, original_tree: URDFTree, group_id: str, final_groups: Dict) -> URDFTree:
        # 1. Initialize the new sub-tree container
        sub_tree = original_tree
        
        # 2. Identify all links that belong to this component sub-graph
        # We trace from every leaf in the group back to the mount point
        group_data = final_groups[group_id]
        component_link_names: Set[str] = set()
        mount_link_name = None

        for leaf_name, info in group_data['details'].items():
            mount_link_name = info['mount'] # The "Attachment Root" from Step 2
            curr = original_tree.links.get(leaf_name)
            while curr and curr.name != mount_link_name:
                component_link_names.add(curr.name)
                curr = curr.parent

        if not component_link_names:
            return sub_tree

        # 3. Create new URDFLink instances for the sub-tree to avoid mutating the original
        for name in component_link_names:
            orig_link = original_tree.links[name]
            new_link = URDFLink(name)
            new_link.visual_count = orig_link.visual_count
            new_link.collision_count = orig_link.collision_count
            new_link.joint_to_parent_type = orig_link.joint_to_parent_type
            new_link.joint_name = orig_link.joint_name
            sub_tree.links[name] = new_link

        # 4. Re-establish parent/child relationships within the sub-tree
        for name, new_link in sub_tree.links.items():
            orig_link = original_tree.links[name]
            
            # If the original parent is in our component set, link them
            if orig_link.parent and orig_link.parent.name in component_link_names:
                parent_in_sub = sub_tree.links[orig_link.parent.name]
                new_link.parent = parent_in_sub
                parent_in_sub.children.append(new_link)
            
            # Identify the local root (the link whose parent is the Mount Link)
            elif orig_link.parent and orig_link.parent.name == mount_link_name:
                sub_tree.root = new_link

        # 5. Populate leaf_links for the new tree
        sub_tree.leaf_links = [l for l in sub_tree.links.values() if not l.children]
        
        return sub_tree
    
    def populate_chain_ontology_new(
        self, urdf_tree: Any, final_chains: Dict[str, Any]
    ) -> Set[str]:
        """
        Populates ontology concepts for chain components, ordered segments, and orphan nodes.
        (Upper-level component connections are deferred).
        """
        all_orphan_nodes: Set[str] = set()

        # 1. Build local set of classified links for this stage
        all_classified_links: Set[str] = set()
        for chain_comp in final_chains.values():
            all_classified_links.update(getattr(chain_comp, "all_links", []))

        for gid, chain_comp in final_chains.items():
            comp_name = getattr(chain_comp, "semantic_name", None) or gid
            mount_name = getattr(chain_comp, "attachment_root", None)
            component_links = list(getattr(chain_comp, "all_links", []))

            if not component_links:
                continue

            # Concepts & Types
            if comp_name:
                self.ontology_handler.feeder.addConcept(comp_name)

            if getattr(chain_comp, "type", None):
                self.ontology_handler.feeder.addInheritage(comp_name, chain_comp.type)

            if mount_name:
                self.ontology_handler.feeder.addDataRelation(comp_name, "mountsOn", "string", mount_name)

            if getattr(chain_comp, "assembly_base_link", None):
                self.ontology_handler.feeder.addDataRelation(comp_name,"hasAssemblyLinkFrame", "string", chain_comp.assembly_base_link)

            # Segments & Kinematic Chain Topology
            segments = getattr(chain_comp, "segments", [])
            ordered_segment_items = []

            if segments:
                for seg in segments:
                    seg_name = (
                        getattr(seg, "semantic_name", None)
                        or getattr(seg, "id", None)
                        or getattr(seg, "link_name", None)
                    )
                    seg_type = getattr(seg, "type", "ChainSegment")
                    link_frame = getattr(seg, "link_name", seg_name)
                    ordered_segment_items.append((seg_name, seg_type, link_frame))
            else:
                for link_name in component_links:
                    ordered_segment_items.append(
                        (link_name, "ChainSegment", link_name)
                    )

            for seg_name, seg_type, link_frame in ordered_segment_items:
                self.ontology_handler.feeder.addConcept(seg_name)
                self.ontology_handler.feeder.addInheritage(seg_name, seg_type)
                self.ontology_handler.feeder.addDataRelation(seg_name, "hasLinkFrame", "string", link_frame)

            if ordered_segment_items:
                first_seg_name = ordered_segment_items[0][0]
                last_seg_name = ordered_segment_items[-1][0]

                self.ontology_handler.feeder.addObjectRelation(comp_name, "hasInitialSegment", first_seg_name)

                for i in range(len(ordered_segment_items) - 1):
                    curr_seg_name = ordered_segment_items[i][0]
                    next_seg_name = ordered_segment_items[i + 1][0]
                    self.ontology_handler.feeder.addObjectRelation( curr_seg_name, "hasNextSegment", next_seg_name)

                self.ontology_handler.feeder.addObjectRelation(comp_name, "hasTerminalSegment", last_seg_name)

            # 2. Discover Orphans Attached to Chain Links
            for link_name in component_links:
                if link_name not in urdf_tree.links:
                    continue

                comp_link_obj = urdf_tree.links[link_name]
                external_children = [
                    child
                    for child in comp_link_obj.children
                    if child.name not in all_classified_links
                ]

                for orphan_entry_link in external_children:
                    self._process_and_register_orphan_subtree(
                        current_link=orphan_entry_link,
                        parent_comp_name=comp_name,
                        all_classified_links=all_classified_links,
                        orphan_set=all_orphan_nodes,
                    )

        print(
            f"[ONTOLOGY CHAIN SUCCESS] Created concepts for chain components, segments, and {len(all_orphan_nodes)} orphan nodes."
        )
        return all_orphan_nodes
        
    def populate_ontology_tree_new(
        self, urdf_tree: Any, final_trees: Dict[str, Any]
    ) -> Set[str]:
        """
        Populates ontology concepts and relations for tree assemblies,
        sub-branches, and orphan nodes. (Upper-level component connections are deferred).
        """
        all_orphan_nodes: Set[str] = set()

        # 1. Build local set of classified links for this stage
        all_classified_links: Set[str] = set()
        for tree_comp in final_trees.values():
            all_classified_links.update(getattr(tree_comp, "all_links", []))

        for gid, tree_comp in final_trees.items():
            comp_name = getattr(tree_comp, "semantic_name", None) or gid
            mount_name = getattr(tree_comp, "attachment_root", None)
            component_links = set(getattr(tree_comp, "all_links", []))

            # Concepts & Types
            if comp_name:
                self.ontology_handler.feeder.addConcept(comp_name)

            if getattr(tree_comp, "type", None):
                self.ontology_handler.feeder.addInheritage(comp_name, tree_comp.type)
                print("ASSERTION:", comp_name, "isA", tree_comp.type)

            if mount_name:
                self.ontology_handler.feeder.addDataRelation(comp_name, "mountsOn", "string", mount_name)
                print("ASSERTION:", comp_name, "mountsOn", mount_name)

            if getattr(tree_comp, "assembly_base_link", None):
                self.ontology_handler.feeder.addDataRelation(comp_name, "hasAssemblyLinkFrame", "string", tree_comp.assembly_base_link)
                print("ASSERTION:", comp_name, "hasAssemblyLinkFrame", tree_comp.assembly_base_link)

            # Sub-branches
            branches = getattr(tree_comp, "branches", [])
            if len(branches) == 1 and len(getattr(branches[0], "link_chain", [])) == 1:
                print(f"[SKIP SUB-PARTS] Single-branch tree component: {comp_name}")
            else:
                for sub_comp in branches:
                    sub_comp_name = getattr(
                        sub_comp, "semantic_name", None
                    ) or getattr(sub_comp, "id", "sub_part")
                    self.ontology_handler.feeder.addConcept(sub_comp_name)
                    self.ontology_handler.feeder.addInheritage(sub_comp_name, getattr(sub_comp, "type", "Subpart"))
                    print("ASSERTION:", sub_comp_name, "isA", sub_comp.type)
                    self.ontology_handler.feeder.addObjectRelation(comp_name, "hasSubPart", sub_comp_name)
                    print("ASSERTION:", comp_name, "hasSubPart", sub_comp_name)
                    for link in getattr(sub_comp, "link_chain", []):
                        self.ontology_handler.feeder.addDataRelation(sub_comp_name, "hasLinkFrame", "string", link)
                        

            # 2. Discover Orphans Attached to Tree Links
            for link_name in component_links:
                if link_name not in urdf_tree.links:
                    continue

                comp_link_obj = urdf_tree.links[link_name]
                external_children = [
                    child
                    for child in comp_link_obj.children
                    if child.name not in all_classified_links
                ]

                for orphan_entry_link in external_children:
                    self._process_and_register_orphan_subtree(
                        current_link=orphan_entry_link,
                        parent_comp_name=comp_name,
                        all_classified_links=all_classified_links,
                        orphan_set=all_orphan_nodes,
                    )

        print(
            f"[ONTOLOGY TREE SUCCESS] Created concepts for tree components and {len(all_orphan_nodes)} orphan nodes."
        )
        return all_orphan_nodes
    
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

            # Retrieve parent component ID resolved by resolve_component_hierarchy
            parent_comp_id = (
                comp_data.get("parent_component_id")
                if is_dict
                else getattr(comp_data, "parent_component_id", None)
            )

            # CASE 1: Child Component -> Has a valid parent component in the assembly
            if parent_comp_id and parent_comp_id in resolved_components:
                parent_data = resolved_components[parent_comp_id]
                parent_name = (
                    (
                        parent_data.get("semantic_name")
                        if isinstance(parent_data, dict)
                        else getattr(parent_data, "semantic_name", None)
                    )
                    or parent_comp_id
                )

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

        # 1. Map every link frame -> owning component ID
        for comp_id, comp_data in all_components.items():
            all_links = (
                comp_data.get("all_links", [])
                if isinstance(comp_data, dict)
                else getattr(comp_data, "all_links", [])
            )
            for link_name in all_links:
                link_owner_map[link_name] = comp_id

        # 2. Resolve parent component for each entity
        for comp_id, comp_data in all_components.items():
            if isinstance(comp_data, dict):
                mount_link = comp_data.get("mount_link") or comp_data.get("attachment_root")
            else:
                mount_link = getattr(comp_data, "mount_link", None) or getattr(comp_data, "attachment_root", None)

            parent_comp_id = link_owner_map.get(mount_link) if mount_link else None

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
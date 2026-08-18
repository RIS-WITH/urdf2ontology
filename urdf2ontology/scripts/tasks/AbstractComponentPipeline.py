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
    def __init__(self, keyword, nb_runs, ontology_handler):
        self.keyword = keyword
        self.nb_runs = nb_runs
        self.ontology_handler = ontology_handler
        self.llm_module = LLMInterface("gpt-oss:20b", 0.7) #"gpt-oss:20b", "magistral:24b", "qwen3.6:27b", "gemma3:27b"
   
    # # initial function
    # def run_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, full_verbose = False, export_result = False):
    #     self.init_urdf = urdf_tree
        
    #     # print(urdf_tree.render_tree(start_link="narrow_stereo_link"))
    #     # print(urdf_tree.render_tree(start_link="narrow_stereo_link", show_joint_info=True))
    #     robot_leaf_links = urdf_tree.get_leaf_links()
    #     leaf_links_str = [leaf.name for leaf in robot_leaf_links]
    #     robot_leaf_links_dict = {'links': leaf_links_str}

    #     print(f"<==== 1. IDENTIFICATION TASK ====>") # no need for joint types and names as we gather entry points
    #     component_identif = ComponentLinkIdentificationTask(self.nb_runs, self.keyword)
    #     result_identif = self.llm_module.run_task(component_identif, robot_leaf_links_dict, verbose=full_verbose)
    #     print(f"Final Link identification task: {result_identif}")

    #     robot_components.add_identified_component_links(result_identif)
        
    #     print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>") 
    #     component_mounting = ComponentLinkMountingPointTask(self.nb_runs, self.keyword)
    #     result_mounting = self.llm_module.run_task(component_mounting, robot_components.initial_chains, verbose=full_verbose)
    #     print(f"Final Link attachment root detection task: {result_mounting}")
        
    #     robot_components.add_trimmed_component_chains(result_mounting)
     
    #     print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
    #     pre_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword)

    #     upper_concept = pre_classification_task.parent_context + "Subpart"
    #     init_link_states = {name: upper_concept for name in robot_components.trimmed_chains.keys()}
    #     link_states = self.run_iterative_classification(pre_classification_task, init_link_states, robot_components.trimmed_chains, 5, verbose = full_verbose)
    #     print(f"Final Pre-Classification Task: {link_states}")
        
    #     robot_components.add_preclassif(link_states)
        
    #     print(f"<==== 4. GROUPING TASK ====>")
    #     component_grouping = ComponentLinkGroupingTask(self.nb_runs, self.keyword)
        
    #     result_grouping = self.llm_module.run_task(component_grouping, robot_components.trimmed_chains, verbose=True)
    #     print(f"Final Link grouping task: {result_grouping}")
        
    #     robot_components.add_component_groups(result_grouping, self.keyword)
    
    #     robot_components.build_component_groups(robot_components.trimmed_chains, 
    #                                             robot_components.component_preclassif, 
    #                                             robot_components.component_groups,
    #                                             self.keyword)
        
    #     print(f"<==== 5. POST-CLASSIFICATION TASK ====>")
    #     post_classification_task = ComponentLinkGroupClassificationTask(self.nb_runs, self.keyword)
        
    #     upper_concept = post_classification_task.parent_context
    #     init_group_states = {gid: upper_concept for gid in robot_components.final_groups.keys()}
    #     group_states = self.run_iterative_classification(post_classification_task, init_group_states, robot_components.final_groups, 5, verbose = full_verbose)
    #     print(f"Final Post-Classification Task: {group_states}")
        
    #     robot_components.add_postclassif(group_states)
        
    #     print(" =================== BUILDING FINAL GROUPS =================")
    #     final_groups = robot_components.build_final_group_mapping(robot_components.final_groups, robot_components.component_postclassif)
    #     print(f"Final Link Groups: {final_groups}")
        
    #     # # ============== Ontology Population & verification ============
    #     print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
    #     self.populate_ontology_groups(final_groups)
        
    #     # ======== unused
    #     # GET THE URDF SUB-TREE EMBEDDING EACH GROUP
    #     # for group_id in final_groups.keys():
    #     #     sub_tree = self.extract_component_as_tree(self.init_urdf, group_id, final_groups=final_groups)
    #     #     print(sub_tree.print_tree())
    #     #     robot_components.urdf_groups.update({group_id: sub_tree})
        
    #     urdf_tree.abstract_components(final_groups)
        
    #     if(export_result):
    #         robot_name = self.init_urdf.urdf_path.split('/')[-1].split('.')[0]
    #         path = f"/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/results/log_{robot_name}_{self.keyword}.json"
    #         with open(path, 'w') as f:
    #             json.dump(final_groups, f, indent=2)
        
    #     return urdf_tree
    
    # top-down
    def run_chain_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, full_verbose: bool = False, save_intermediate: bool = False, load_intermediate: bool = False) :
        # Used for bodyparts abstraction
        robot_leaf_links = urdf_tree.get_leaf_links()
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}     
        
        user_input = {f"link_{i}": link for i, link in enumerate(robot_leaf_links_dict["links"])}
        
        if(load_intermediate):
            path = "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology"
            robot_components.components = np.load(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
        else:
            print(f"<==== 1. IDENTIFICATION TASK ====>")
            print("Initial Link Input:", robot_leaf_links_dict.get('links'))
            component_identif = ComponentLinkIdentificationTask(self.nb_runs*3, self.keyword)
            # result_identif = self.llm_module.run_task(component_identif, robot_leaf_links_dict, verbose=full_verbose, verbose_thinking = False)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=10, verbose=True)
            print(f"Final Link identification task: {result_identif}")
            print(f"Final Link identification task: {result_identif}")
            
            robot_components.create_component_links(result_identif, "chain")

            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>") # requires several runs (minimum 3) -> batch
            component_mounting = ComponentLinkMountingPointTask(self.nb_runs*3, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                                batch_size = 5, verbose=full_verbose, verbose_thinking = False)
            # version witout batching
            # result_mounting = self.llm_module.run_task(component_mounting, robot_components.components, verbose=verbose, verbose_thinking = False)
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
            segmentation_task = SegmentIdentificationTask(self.nb_runs*3, self.keyword, "links")
            segments_output = self.llm_module.run_task(segmentation_task, robot_components.components, verbose = True, verbose_thinking = False)
            print(segments_output)
            
            #robot_components.create_segments_chain(segments_output)
            
            segments = robot_components.create_segments_chain_canonical(segments_output)
            
            # print(segments)
            for comp_id, component in robot_components.components.items():
                if component.segments:
                    # Extract segment IDs in kinematic order
                    # full_list = []
                    # for seg in component.segments:
                    #     for link in seg.link_chain:
                    #         full_list.append(link)
                    segment_ids = [seg.id for seg in component.segments]
                    
                    # Format into sequential text
                    sequence_str = " then ".join(segment_ids)
                    print(f"{comp_id} is composed of {sequence_str}")
                    
            # TO DO: hide the segment state dict (into a flattened dict?)
            flat_segments = {seg.id: seg for chain in robot_components.components.values() for seg in chain.segments}
            upper_concept = "BodyPartSubpart"
            init_segment_states = {seg.id: upper_concept for seg in flat_segments.values()}
            
            # print("INIT SEGMENT STATES : ", init_segment_states)
            # TO DO: remove duplicates before classification (e.g., [base_link, torso])
            # generate a set of tuples for each subchain (sub_component)
            
            print(f"<==== 5. POST-CLASSIFICATION SEGMENTS TASK ====>")
            post_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword, "links")
            #upper_concept = post_classification_task.parent_context + "BodyPart"        
            segment_states = self.run_iterative_classification(post_classification_task, init_segment_states, flat_segments, 3, verbose = full_verbose)
            print(f"Final Post-Classification Task: {segment_states}")
            
            robot_components.add_postclassification_segments(segment_states)
                    
            # print(f"<==== 6. SEGMENT GROUPING TASK ====>")
            # component_grouping = ComponentLinkGroupingTask(self.nb_runs, self.keyword, "links")
            # result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=full_verbose, verbose_thinking = False)
            # print(f"Final Link grouping task: {result_grouping}")
            
            print(f"<==== 6. CHAIN GROUPING TASK ====>")
            component_grouping = ChainLinkGroupingTask(self.nb_runs, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=True, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            
            value = robot_components.build_macro_entities_dict(result_grouping)
            print(value)
            for elem in value:
                print(elem)
                
            #robot_components.create_grouped_components(result_grouping, self.keyword)

            print(f"<==== 6. LINK NAMING TASK ====>")
            # # first string corresponds to the type of entity to name, the second one to the type of input chain to provide
            component_link_naming = ComponentLinkNamingTask(self.nb_runs, self.keyword, "chain_component", "links")
            result_link_naming = self.llm_module.run_task(component_link_naming, robot_components.components, verbose=full_verbose, verbose_thinking=False)
            print(f"Final Link Naming task: {result_link_naming}")
            
            robot_components.assign_semantic_names(result_link_naming, "chain")
            
            # =============== INTERMEDIARY SAVE CHECKPOINT ============= 
            #np.save("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/tmp.npy", robot_components.bodypart_components)
            #robot_components.bodypart_components =  np.load("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/tmp.npy", allow_pickle= True)
            if(save_intermediate):
                np.save(f"/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/eval/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.components)
                robot_components.components =  np.load(f"/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/eval/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
            
                
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
        report_path = f"/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/eval/timing_{urdf_tree.robot_name}_{self.keyword}.json"
        self.llm_module.save_metrics(report_path)
        
        return urdf_tree
    
    # bottom-up
    def run_tree_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, verbose: bool = False, save_intermediate: bool = False, load_intermediate: bool = False) :
        self.init_urdf = urdf_tree
        # print(urdf_tree.render_tree(start_link="narrow_stereo_link"))
        # print(urdf_tree.render_tree(start_link="narrow_stereo_link", show_joint_info=True))
        robot_leaf_links = urdf_tree.get_leaf_links()
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}
        
        user_input = {f"link_{i}": link for i, link in enumerate(robot_leaf_links_dict["links"])}
               
        if(load_intermediate):
            path = "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology"
            robot_components.merged_components = np.load(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", allow_pickle= True).item()
        else:

            print(f"<==== 1. IDENTIFICATION TASK ====>")
            component_identif = ComponentLinkIdentificationTask(self.nb_runs*5, self.keyword)
            #result_identif = self.llm_module.run_task(component_identif, robot_leaf_links_dict, verbose=True)
            result_identif = self.llm_module.run_batched_task(component_identif, user_input, batch_size=10, verbose=True)
            print(f"Final Link identification task: {result_identif}")

            robot_components.create_component_links(result_identif, "branch")
                         
            print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>") # requires several runs (minimum 3) -> batch
            component_mounting = ComponentLinkMountingPointTask(self.nb_runs*5, self.keyword, "kinematic_csv")
            result_mounting = self.llm_module.run_batched_task(component_mounting, robot_components.components, 
                                                               batch_size = 5, verbose=verbose, verbose_thinking = False)
            # version witout batching
            # result_mounting = self.llm_module.run_task(component_mounting, robot_components.components, verbose=verbose, verbose_thinking = False)
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
        
            print(f"<==== 4. GROUPING TASK ====>") # requires several runs (minimum 3)
            component_grouping = ComponentLinkGroupingTask(self.nb_runs*5, self.keyword, "links")
            result_grouping = self.llm_module.run_task(component_grouping, robot_components.components, verbose=verbose, verbose_thinking = False)
            print(f"Final Link grouping task: {result_grouping}")
            
            robot_components.create_grouped_components(result_grouping, self.keyword)
            
            print(f"<==== 5. POST-CLASSIFICATION TASK ====>")
            post_classification_task = ComponentLinkGroupClassificationTask(self.nb_runs, self.keyword, "links")
            upper_concept = post_classification_task.parent_context
            init_group_states = {gid: upper_concept for gid in robot_components.merged_components.keys()}
            group_states = self.run_iterative_classification(post_classification_task, init_group_states, robot_components.merged_components, 
                                                             max_iterations = 3, verbose = verbose)
            print(f"Final Post-Classification Task: {group_states}")
            
            robot_components.add_postclassification(group_states)
        
            print(f"<==== 6. LINK NAMING TASK ====>")
            component_link_naming = ComponentLinkNamingTask(self.nb_runs, self.keyword, "tree_component", "links")
            result_link_naming = self.llm_module.run_task(component_link_naming, robot_components.merged_components, verbose=True, verbose_thinking=False)
            print(f"Final Link Naming task: {result_link_naming}")
            
            robot_components.assign_semantic_names(result_link_naming, "tree")
        
        # OPTIONAL CHECKPOINT IN NUMPY ARRAY
        if(save_intermediate):
            path = "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/eval/"
            np.save(path + f"/{urdf_tree.robot_name}_tmp_{self.keyword}.npy", robot_components.merged_components)

        # print(f"<==== 6. LINK NAMING TASK ====>")
        # component_link_naming = ComponentLinkNamingTask(self.nb_runs, self.keyword, "tree_component", "links")
        # result_link_naming = self.llm_module.run_task(component_link_naming, robot_components.merged_components, verbose=True, verbose_thinking=False)
        # print(f"Final Link Naming task: {result_link_naming}")
        
        # robot_components.assign_semantic_names(result_link_naming, "tree")
            
        # ============== Ontology Population & verification ============
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
        #self.populate_ontology_tree_groups(robot_components.merged_components, False)
        #urdf_tree.abstract_components_tree_safe(robot_components.merged_components)
        
        # GET THE URDF SUB-TREE EMBEDDING EACH GROUP
        # for group_id in final_groups.keys():
        #     sub_tree = self.extract_component_as_tree(self.init_urdf, group_id, final_groups=final_groups)
        #     print(sub_tree.print_tree())
        #     robot_components.urdf_groups.update({group_id: sub_tree})
        
        # 1. Populate Ontology 
        orphan_links = self.populate_ontology_tree(urdf_tree=urdf_tree, final_trees=robot_components.merged_components)

        # 2. Purge URDF Tree Structure
        urdf_tree.purge_abstracted_components(final_trees=robot_components.merged_components, orphan_links=orphan_links)
        
        for comp in robot_components.merged_components.values():
            print(comp)
            
        # removed_topology_mask = urdf_tree.abstract_components_tree_v4(robot_components.merged_components, True)
        # print(removed_topology_mask)
        # self.populate_ontology_and_abstract_trees(robot_components.merged_components)
        
        # urdf_tree.abstract_tree_components_v3(robot_components.merged_components, False)
        report_path = f"/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/eval/timing_{urdf_tree.robot_name}_{self.keyword}.json"
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
        if(save_intermediate):
            robot_name = self.init_urdf.urdf_path.split('/')[-1].split('.')[0]
            self.ontology_handler.actions.save("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/results/abstraction_" + robot_name +  "_" + self.keyword +".owl") 
            
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
                    
            # if(save_intermediate):
            #     robot_name = self.init_urdf.urdf_path.split('/')[-1].split('.')[0]
            #     self.ontology_handler.actions.save("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/results/abstraction_" + robot_name +  "_" + self.keyword +".owl") 
     
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
                self.ontology_handler.feeder.addObjectRelation(comp_name, "mountsOn", mount_name)

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
        
    def populate_ontology_tree(
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
                self.ontology_handler.feeder.addObjectRelation(comp_name, "mountsOn", mount_name)
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
        self.ontology_handler.feeder.addDataRelation(link_name, "hasLinkFrame", link_name)

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
                
    def resolve_and_assert_component_hierarchy(self, all_components: Dict[str, Any], robot_individual_name: str) -> None:
        """
        Final Post-Processing Pass:
        Resolves inter-component hierarchy across all extracted steps and asserts
        'hasSubSystem' relations (Parent Component -> hasSubSystem -> Child Component).
        """
        # Resolve inter-component hierarchy in memory across all components
        resolved_components = self.resolve_component_hierarchy(all_components)

        robot_instance_name = robot_individual_name + "_robot"
        self.ontology_handler.feeder.addConcept(robot_instance_name)
            
        relations_count = 0
        for comp_id, comp_data in resolved_components.items():
            is_dict = isinstance(comp_data, dict)

            # Retrieve parent component ID resolved by resolve_component_hierarchy
            parent_comp_id = (
                comp_data.get("parent_component_id")
                if is_dict
                else getattr(comp_data, "parent_component_id", None)
            )

            if parent_comp_id and parent_comp_id in resolved_components:
                child_name = (
                    comp_data.get("semantic_name")
                    if is_dict
                    else getattr(comp_data, "semantic_name", None)
                ) or comp_id

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
                self.ontology_handler.feeder.addObjectRelation(parent_name, "hasSubSystem", child_name)
                print("ASSERTION:", parent_name, "hasSubSystem", child_name)
                self.ontology_handler.feeder.addObjectRelation(robot_instance_name, "part", parent_name)
                print("ASSERTION:", robot_instance_name, "part", parent_name)
                relations_count += 1

        print(
            f"\n[HIERARCHY RESOLVED] Asserted {relations_count} 'hasSubSystem' relations in ontology."
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
    
        
    # def populate_hierarchical_ontology(self, all_components: Dict[str, Any] ) -> Set[str]:
    #     """
    #     Main entry point for populating the entire robot ontology.
    #     Resolves inter-component hierarchy, dispatches to chain/tree population,
    #     and wires component-to-component parent relations.
    #     """
    #     all_classified_links: Set[str] = set()

    #     # Step 1: Wire component-to-component hierarchy in memory
    #     resolved_components = self.resolve_component_hierarchy(all_components)

    #     # Step 2: Track all classified links across all components
    #     for comp in resolved_components.values():
    #         links = comp.get("all_links", []) if isinstance(comp, dict) else getattr(comp, "all_links", [])
    #         all_classified_links.update(links)

    #     # Step 3: Populate internal ontology for each component
    #     for comp_id, comp_data in resolved_components.items():
            
    #         # Determine if this component is a Tree or a Chain
    #         has_branches = (
    #             ("branches" in comp_data) if isinstance(comp_data, dict) 
    #             else hasattr(comp_data, "branches")
    #         )

    #         if has_branches:
    #             self.populate_tree_ontology(comp_id, comp_data)
    #         else:
    #             self.populate_chain_ontology(comp_id, comp_data)

    #         # Step 4: Add INTER-COMPONENT hierarchy relation (attachedToComponent)
    #         if isinstance(comp_data, dict):
    #             comp_name = comp_data.get("semantic_name") or comp_id
    #             parent_comp_id = comp_data.get("parent_component_id")
    #         else:
    #             comp_name = getattr(comp_data, "semantic_name", None) or comp_id
    #             parent_comp_id = getattr(comp_data, "parent_component_id", None)

    #         if parent_comp_id:
    #             parent_data = resolved_components[parent_comp_id]
    #             parent_name = (
    #                 parent_data.get("semantic_name") or parent_comp_id
    #                 if isinstance(parent_data, dict)
    #                 else getattr(parent_data, "semantic_name", None) or parent_comp_id
    #             )
    #             # Asserts component-to-component link
    #             self.ontology_handler.feeder.addObjectRelation(
    #                 comp_name, "attachedToComponent", parent_name
    #             )

    #     return all_classified_links

    # def populate_chain_ontology(self, gid: str, chain_comp: Any) -> None:
    #     """
    #     Populates ontology assertions for sequential kinematic chains and segment topology.
    #     """
    #     is_dict = isinstance(chain_comp, dict)
        
    #     comp_name = (chain_comp.get("semantic_name") if is_dict else getattr(chain_comp, "semantic_name", None)) or gid
    #     mount_name = chain_comp.get("attachment_root") if is_dict else getattr(chain_comp, "attachment_root", None)
    #     comp_type = (chain_comp.get("type") or chain_comp.get("macro_type")) if is_dict else (getattr(chain_comp, "type", None) or getattr(chain_comp, "macro_type", None))
    #     assembly_base_link = chain_comp.get("assembly_base_link") if is_dict else getattr(chain_comp, "assembly_base_link", None)
    #     component_links = list(chain_comp.get("all_links", [])) if is_dict else list(getattr(chain_comp, "all_links", []))

    #     if not component_links:
    #         return

    #     # =====================================================================
    #     # STEP 1: CHAIN COMPONENT CONCEPT & ROOT RELATIONS
    #     # =====================================================================
    #     if comp_name:
    #         self.ontology_handler.feeder.addConcept(comp_name)

    #     if comp_type:
    #         self.ontology_handler.feeder.addInheritage(comp_name, comp_type)

    #     if mount_name:
    #         self.ontology_handler.feeder.addObjectRelation(comp_name, "mountsOn", mount_name)

    #     if assembly_base_link:
    #         self.ontology_handler.feeder.addDataRelation( comp_name, "hasAssemblyLinkFrame", "string", assembly_base_link )

    #     # =====================================================================
    #     # STEP 2: SEGMENTS & KINEMATIC CHAIN TOPOLOGY
    #     # =====================================================================
    #     segments = chain_comp.get("segments", []) if is_dict else getattr(chain_comp, "segments", [])
    #     ordered_segment_items = []

    #     if segments:
    #         for seg in segments:
    #             s_is_dict = isinstance(seg, dict)
    #             seg_name = (
    #                 (seg.get("semantic_name") or seg.get("id") or seg.get("link_name"))
    #                 if s_is_dict
    #                 else (getattr(seg, "semantic_name", None) or getattr(seg, "id", None) or getattr(seg, "link_name", None))
    #             )
    #             seg_type = seg.get("type", "ChainSegment") if s_is_dict else getattr(seg, "type", "ChainSegment")
    #             link_frame = seg.get("link_name", seg_name) if s_is_dict else getattr(seg, "link_name", seg_name)
    #             ordered_segment_items.append((seg_name, seg_type, link_frame))
    #     else:
    #         # Fallback to component_links order if explicit segment objects aren't present
    #         for link_name in component_links:
    #             ordered_segment_items.append((link_name, "ChainSegment", link_name))

    #     # 1. Create individual concept & link frame for each segment
    #     for seg_name, seg_type, link_frame in ordered_segment_items:
    #         self.ontology_handler.feeder.addConcept(seg_name)
    #         self.ontology_handler.feeder.addInheritage(seg_name, seg_type)
    #         self.ontology_handler.feeder.addDataRelation(seg_name, "hasLinkFrame", "string", link_frame)

    #     # 2. Wire sequential chain properties in the ontology
    #     if ordered_segment_items:
    #         first_seg_name = ordered_segment_items[0][0]
    #         last_seg_name = ordered_segment_items[-1][0]

    #         # Component -> hasInitialSegment -> First Segment
    #         self.ontology_handler.feeder.addObjectRelation(comp_name, "hasInitialSegment", first_seg_name
    #         )

    #         # Segment_i -> hasNextSegment -> Segment_{i+1}
    #         for i in range(len(ordered_segment_items) - 1):
    #             curr_seg_name = ordered_segment_items[i][0]
    #             next_seg_name = ordered_segment_items[i + 1][0]
    #             self.ontology_handler.feeder.addObjectRelation(curr_seg_name, "hasNextSegment", next_seg_name)

    #         # Component -> hasTerminalSegment -> Last Segment
    #         self.ontology_handler.feeder.addObjectRelation( comp_name, "hasTerminalSegment", last_seg_name)

    # def populate_tree_ontology(self, gid: str, tree_comp: Any) -> None:
        """
        Populates ontology assertions for tree assemblies and sub-branches.
        """
        is_dict = isinstance(tree_comp, dict)

        comp_name = (tree_comp.get("semantic_name") if is_dict else getattr(tree_comp, "semantic_name", None)) or gid
        mount_name = tree_comp.get("attachment_root") if is_dict else getattr(tree_comp, "attachment_root", None)
        comp_type = (tree_comp.get("type") or tree_comp.get("macro_type")) if is_dict else (getattr(tree_comp, "type", None) or getattr(tree_comp, "macro_type", None))
        assembly_base_link = tree_comp.get("assembly_base_link") if is_dict else getattr(tree_comp, "assembly_base_link", None)

        # =====================================================================
        # STEP 1: TREE COMPONENT CONCEPT & ROOT RELATIONS
        # =====================================================================
        if comp_name:
            self.ontology_handler.feeder.addConcept(comp_name)

        if comp_type:
            self.ontology_handler.feeder.addInheritage(comp_name, comp_type)

        if mount_name:
            self.ontology_handler.feeder.addObjectRelation(comp_name, "mountsOn", mount_name)

        if assembly_base_link:
            self.ontology_handler.feeder.addDataRelation(comp_name, "hasAssemblyLinkFrame", "string", assembly_base_link)

        # =====================================================================
        # STEP 2: SUB-BRANCHES & TREE TOPOLOGY
        # =====================================================================
        branches = tree_comp.get("branches", []) if is_dict else getattr(tree_comp, "branches", [])

        # Check for single-branch single-link edge case
        first_branch_chain = (
            (branches[0].get("link_chain", []) if isinstance(branches[0], dict) else getattr(branches[0], "link_chain", []))
            if branches else []
        )
        if len(branches) == 1 and len(first_branch_chain) == 1:
            print(f"[SKIP SUB-PARTS] Single-branch tree component: {comp_name}")
        else:
            for sub_comp in branches:
                b_is_dict = isinstance(sub_comp, dict)
                
                sub_comp_name = (
                    (sub_comp.get("semantic_name") or sub_comp.get("id"))
                    if b_is_dict
                    else (getattr(sub_comp, "semantic_name", None) or getattr(sub_comp, "id", "sub_part"))
                ) or "sub_part"

                sub_comp_type = sub_comp.get("type", "Subpart") if b_is_dict else getattr(sub_comp, "type", "Subpart")
                link_chain = sub_comp.get("link_chain", []) if b_is_dict else getattr(sub_comp, "link_chain", [])

                self.ontology_handler.feeder.addConcept(sub_comp_name)
                self.ontology_handler.feeder.addInheritage(sub_comp_name, sub_comp_type)
                self.ontology_handler.feeder.addObjectRelation(comp_name, "hasSubPart", sub_comp_name)

                for link in link_chain:
                    self.ontology_handler.feeder.addDataRelation(sub_comp_name, "hasLinkFrame", "string", link)
        
    # def populate_hierarchical_ontology(self, urdf_tree: Any, all_components: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Orchestrates full ontology population: resolves component hierarchy, dispatches
    #     tree/chain population, and links parent-child relationships.
    #     """
    #     # Step 1: Wire inter-component parent/child IDs
    #     resolved_components = self.resolve_component_hierarchy(all_components)

    #     # Step 2: Populate each component's internal structure and inter-component relations
    #     for comp_id, comp_data in resolved_components.items():
            
    #         # --- A. Dispatch based on component structural type ---
    #         has_branches = (
    #             ("branches" in comp_data) if isinstance(comp_data, dict) 
    #             else hasattr(comp_data, "branches")
    #         )
            
    #         if has_branches:
    #             self.populate_tree_ontology(comp_data)
    #         else:
    #             self.populate_chain_ontology(comp_data)

    #         # --- B. Add top-level hierarchy & mounting relations ---
    #         if isinstance(comp_data, dict):
    #             comp_name = comp_data.get("semantic_name") or comp_id
    #             parent_comp_id = comp_data.get("parent_component_id")
    #             mount_link = comp_data.get("mount_link") or comp_data.get("attachment_root")
    #         else:
    #             comp_name = getattr(comp_data, "semantic_name", None) or comp_id
    #             parent_comp_id = getattr(comp_data, "parent_component_id", None)
    #             mount_link = getattr(comp_data, "mount_link", None) or getattr(comp_data, "attachment_root", None)

    #         # Link to parent component
    #         if parent_comp_id:
    #             parent_data = resolved_components[parent_comp_id]
    #             parent_name = (
    #                 parent_data.get("semantic_name") or parent_comp_id
    #                 if isinstance(parent_data, dict)
    #                 else getattr(parent_data, "semantic_name", None) or parent_comp_id
    #             )
    #             self.ontology_handler.feeder.addObjectRelation(comp_name, "attachedToComponent", parent_name)

    #         # Link to physical URDF frame
    #         if mount_link:
    #             self.ontology_handler.feeder.addObjectRelation(comp_name, "mountsOnLinkFrame", mount_link)

    #     return resolved_components

    # def populate_tree_ontology(self, comp_data: Any) -> None:
    #     """
    #     Populates internal ontology triples for Tree components and their branches.
    #     """
    #     is_dict = isinstance(comp_data, dict)
    #     comp_id = comp_data.get("entity_id") if is_dict else getattr(comp_data, "entity_id", None)
    #     macro_type = comp_data.get("macro_type") if is_dict else getattr(comp_data, "macro_type", None)
    #     branches = comp_data.get("branches", []) if is_dict else getattr(comp_data, "branches", [])

    #     # 1. Create top-level Tree Component Individual
    #     self.ontology_handler.feeder.addConcept(comp_id)
    #     self.ontology_handler.feeder.addInheritage(comp_id, macro_type)

    #     # 2. Add Branch Sub-elements
    #     for branch in branches:
    #         b_is_dict = isinstance(branch, dict)
    #         branch_id = branch.get("id") if b_is_dict else getattr(branch, "id", None)
    #         branch_type = branch.get("type") if b_is_dict else getattr(branch, "type", None)
    #         link_chain = branch.get("link_chain", []) if b_is_dict else getattr(branch, "link_chain", [])

    #         full_branch_id = f"{comp_id}_{branch_id}"
    #         self.ontology_handler.feeder.addConcept(full_branch_id)
    #         self.ontology_handler.feeder.addInheritage(full_branch_id, branch_type)
    #         self.ontology_handler.feeder.addObjectRelation(comp_id, "hasSubPart", full_branch_id)

    #         for link in link_chain:
    #             self.ontology_handler.feeder.addObjectRelation(full_branch_id, "containsLinkFrame", link)

    # def populate_chain_ontology(self, comp_data: Any) -> None:
        """
        Populates internal ontology triples for Chain components and their segments.
        """
        is_dict = isinstance(comp_data, dict)
        comp_id = comp_data.get("entity_id") if is_dict else getattr(comp_data, "entity_id", None)
        macro_type = comp_data.get("macro_type") if is_dict else getattr(comp_data, "macro_type", None)
        segments = comp_data.get("segments", []) if is_dict else getattr(comp_data, "segments", [])

        # 1. Create top-level Chain Component Individual
        self.ontology_handler.feeder.addIndividual(comp_id, macro_type)

        # 2. Add Segment Sub-elements
        for segment in segments:
            s_is_dict = isinstance(segment, dict)
            seg_id = segment.get("id") if s_is_dict else getattr(segment, "id", None)
            seg_type = segment.get("type") if s_is_dict else getattr(segment, "type", None)
            link_chain = segment.get("link_chain", []) if s_is_dict else getattr(segment, "link_chain", [])

            full_seg_id = f"{comp_id}_{seg_id}"
            self.ontology_handler.feeder.addConcept(full_seg_id)
            self.ontology_handler.feeder.addInheritage(full_seg_id, seg_type)
            self.ontology_handler.feeder.addObjectRelation(comp_id, "hasSegment", full_seg_id)

            for link in link_chain:
                self.ontology_handler.feeder.addObjectRelation(full_seg_id, "containsLinkFrame", link)
from tasks.ComponentLinkIdentificationTask import ComponentLinkIdentificationTask
from tasks.ComponentLinkSingleClassificationTask import ComponentLinkSingleClassificationTask
from tasks.ComponentLinkMountingPointTask import ComponentLinkMountingPointTask 
from tasks.ComponentLinkGroupClassificationTask import ComponentLinkGroupClassificationTask
from tasks.ComponentLinkGroupingTask import ComponentLinkGroupingTask

from urdf_tree import *
from AbstractedRobotLinks import *
from LLMInterface import *


class AbstractComponentPipeline():
    def __init__(self, keyword, nb_runs, definitions_dict, ontology, ontology_handler):
        self.keyword = keyword
        self.nb_runs = nb_runs
        self.definitions_dict = definitions_dict
        self.ontology = ontology
        self.ontology_handler = ontology_handler
        self.llm_module = LLMInterface("gpt-oss:20b", 0.7)
    
    def run_pipeline(self, urdf_tree: URDFTree, robot_components: ComponentLinks, verbose = False):
        self.init_urdf = urdf_tree
        
        # print(urdf_tree.render_tree(start_link="narrow_stereo_link"))
        # print(urdf_tree.render_tree(start_link="narrow_stereo_link", show_joint_info=True))
        robot_leaf_links = urdf_tree.get_leaf_links()
        leaf_links_str = [leaf.name for leaf in robot_leaf_links]
        robot_leaf_links_dict = {'links': leaf_links_str}

        print(f"<==== 1. IDENTIFICATION TASK ====>")
        component_identif = ComponentLinkIdentificationTask(self.nb_runs, self.keyword, self.definitions_dict)
        result_identif = self.llm_module.run_task(component_identif, robot_leaf_links_dict, verbose=False)
        print(f"Final Link identification task: {result_identif}")

        robot_components.add_identified_component_links(result_identif)

        print(f"<==== 2. ATTACHMENT ROOT DETECTION TASK ====>")
        component_mounting = ComponentLinkMountingPointTask(self.nb_runs, self.keyword)
        result_mounting = self.llm_module.run_task(component_mounting, robot_components.initial_chains, verbose=False)
        print(f"Final Link attachment root detection task: {result_mounting}")
        
        robot_components.add_trimmed_component_chains(result_mounting)
        
        print(f"<==== 3. PRE-CLASSIFICATION TASK ====>")
        pre_classification_task = ComponentLinkSingleClassificationTask(self.nb_runs, self.keyword)

        upper_concept = pre_classification_task.parent_context + "Subpart"
        init_link_states = {name: upper_concept for name in robot_components.trimmed_chains.keys()}
        link_states = self.run_iterative_classification(pre_classification_task, init_link_states, robot_components.trimmed_chains, 5)
        print(f"Final Pre-Classification Task: {link_states}")
        
        robot_components.add_preclassif(link_states)

        print(f"<==== 4. GROUPING TASK ====>")
        component_grouping = ComponentLinkGroupingTask(self.nb_runs, self.keyword)
        result_grouping = self.llm_module.run_task(component_grouping, robot_components.trimmed_chains, verbose=False)
        print(f"Final Link grouping task: {result_grouping}")
        
        robot_components.add_component_groups(result_grouping, self.keyword)
    
        robot_components.build_component_groups(robot_components.trimmed_chains, 
                                                robot_components.component_preclassif, 
                                                robot_components.component_groups)
        
        print(f"<==== 5. POST-CLASSIFICATION TASK ====>")
        post_classification_task = ComponentLinkGroupClassificationTask(self.nb_runs, self.keyword)
        
        upper_concept = post_classification_task.parent_context
        init_group_states = {gid: upper_concept for gid in robot_components.final_groups.keys()}
        group_states = self.run_iterative_classification(post_classification_task, init_group_states, robot_components.final_groups, 5)
        print(f"Final Post-Classification Task: {group_states}")
        
        robot_components.add_postclassif(group_states)
        
        print(" =================== BUILDING FINAL GROUPS =================")
        final_groups = robot_components.build_final_group_mapping(robot_components.final_groups, robot_components.component_postclassif)
        print(f"Final Link Groups: {final_groups}")
        
        # ============== Ontology Population & verification ============
        print(f"<==== 7. ONTOLOGY POPULATION TASK ====>")
        for group_id, data in final_groups.items():
            self.ontology_handler.feeder.addConcept(group_id)
            #print("Creating new upper individual :", group_id)
            final_type = data['component_type']
            self.ontology_handler.feeder.addInheritage(group_id, final_type)
            #print("Asserting inheritance over upper individual :", group_id, "isA", final_type)
            
            for link_name in data['links']:
                self.ontology_handler.feeder.addConcept(link_name)
                #print("Creating new lower individual :", link_name)
                # Add the specific subpart role (e.g., 'RgbFrame')
                subpart = data['details'][link_name]['subpart_type']
                self.ontology_handler.feeder.addInheritage(link_name, subpart)
                #print("Asserting inheritance over lower individual :", link_name, "isA", subpart)
                
                # Link the part to the sensor
                self.ontology_handler.feeder.addObjectRelation(group_id, 'hasPart', link_name)
                #print("Asserting property between upper and lower individual :", group_id, "hasPart", link_name)
        
        # robot_name = self.init_urdf.urdf_path.split('/')[-1].split('.')[0]
        # self.ontology_handler.actions.save("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/results/asbtraction_" + robot_name +  "_" + self.keyword +".owl")
        urdf_tree.abstract_components(final_groups)
        
        return urdf_tree
    
    def run_iterative_classification(self, task, initial_states, target_data_map, max_iterations=5):
        """
        Generalizes the iterative LLM refinement logic for both links and groups.
        
        :param task: The LLM task instance (pre_classification_task or post_classification_task)
        :param initial_states: Dict mapping ID to the starting upper concept
        :param target_data_map: Dict containing the actual content (chains or group details)
        :param max_iterations: Max refinement depth
        :return: Final refined states (dict)
        """
        states = initial_states.copy()
        finished_items = set()

        for iteration in range(max_iterations):
            batches = {}
            
            # 1. Group items by their current concept to find valid sub-types
            for item_id, current_concept in states.items():
                if item_id in finished_items:
                    continue
                    
                sub_concepts = self.ontology_handler.classes.getDown(current_concept, 1)
                
                # If no more sub-types exist, this branch of the ontology is exhausted
                if not sub_concepts or len(sub_concepts) == 1:
                    finished_items.add(item_id)
                else:
                    if current_concept not in batches:
                        batches[current_concept] = {"items": {}, "sub_types": sub_concepts}
                    
                    # Logic to handle different input formats for pre vs post tasks
                    if hasattr(task, 'is_post_classification') and task.is_post_classification:
                        batches[current_concept]["items"][item_id] = {
                            "current_category": current_concept,
                            "components": target_data_map[item_id]
                        }
                    else:
                        batches[current_concept]["items"][item_id] = target_data_map[item_id]

            if not batches:
                break

            changes_found = False

            # 2. Run the LLM task for each batch
            for parent_concept, batch_data in batches.items():
                task.parent_context = parent_concept
                task.valid_types = batch_data["sub_types"]
                
                print(f"Iteration {iteration}: Refining {list(batch_data['items'].keys())} under {parent_concept}")
                results = self.llm_module.run_task(task, batch_data["items"], verbose=False)

                for item_id, new_type in results.items():
                    old_type = states[item_id]
                    
                    # Update state only if a valid, new sub-type was found
                    if new_type in batch_data["sub_types"] and new_type != old_type:
                        states[item_id] = new_type
                        changes_found = True
                    else:
                        # If LLM returns parent type or invalid type, we stop refining this item
                        finished_items.add(item_id)

            if not changes_found:
                break

        return states
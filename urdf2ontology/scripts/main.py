#!/usr/bin/env python3
import rclpy
from urdf_tree import URDFTree
from AbstractedRobotLinks import AbstractedRobotLinks
from tasks.AbstractComponentPipeline import AbstractComponentPipeline

from component_ontologies import *
import os
import copy

from ontopy import OntologyManipulator


if __name__ == '__main__':
    
    # DEFINES PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "new_evaluation"))
    ONTOLOGY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ontology"))
    URDF_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "urdf_models"))
        
    rclpy.init()
    ontology_handler = OntologyManipulator()
    
    # Load your OWL files
    files = ["component_ontology.owl"]
    for f in files:
        ontology_handler.actions.fadd(ONTOLOGY_DIR + "/" + f)
    ontology_handler.close()
    
    # URDF models to be evaluated
    eval_models = ['iCubGazeboV2_10.urdf', 'ingenuity.urdf', 'rbvogui_plus.urdf', 'pr2_standard.urdf', 'solo.urdf']

    # Select the index of the model to abstract
    robot_path_string = URDF_DIR + "/" + eval_models[5]
    
    robot_name = robot_path_string.split('/')[-1].split('.')[0]

    urdf_tree = URDFTree(robot_name, robot_path_string)
    
    # Maintains a dict for each of the links in the URDF to keep track of which have been processed
    all_links = [link.name for link in urdf_tree.get_all_links()]
    print("ALL LINKS: ", len(all_links))
    print("init leaf links: ", [link.name for link in urdf_tree.leaf_links])
    
    # For full abstraction
    all_components = {}
    
    path_dict = {"main_path": SCRIPT_DIR, 
                 "eval": EVAL_DIR}
    
    abstracted_robot = AbstractedRobotLinks(robot_name, robot_path_string, urdf_tree)
    init_urdf = abstracted_robot.urdf_tree
    
    print(f"\n================ [SENSOR PASS] ================")
    sensor_pipeline = AbstractComponentPipeline(keyword = "sensor", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)
    sensor_less_urdf = sensor_pipeline.run_tree_pipeline(init_urdf, abstracted_robot.sensor_links, verbose = True, 
                                                         save_intermediate = False, 
                                                         load_intermediate = False)
    all_components.update(abstracted_robot.sensor_links.merged_components)
    
    print("Remaining leaves:", [link.name for link in sensor_less_urdf.leaf_links])

    print(f"\n================ [END EFFECTOR PASS] ================")
    end_effector_pipeline = AbstractComponentPipeline(keyword = "end_effector", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)
    end_effector_less_urdf = end_effector_pipeline.run_tree_pipeline(sensor_less_urdf, abstracted_robot.end_effector_links, verbose = True,
                                                                     save_intermediate = False, 
                                                                     load_intermediate = False)
    all_components.update(abstracted_robot.end_effector_links.merged_components)
    print("Remaining leaves:", [link.name for link in end_effector_less_urdf.leaf_links])

    print(f"\n================ [LOCOMOTION PASS] ================")
    locomotion_pipeline = AbstractComponentPipeline(keyword = "locomotion", nb_runs = 1, ontology_handler= ontology_handler, paths = path_dict)
    locomotion_less_urdf = locomotion_pipeline.run_tree_pipeline(end_effector_less_urdf, abstracted_robot.actuator_links, verbose = True,
                                                                save_intermediate = False, 
                                                                load_intermediate = False)
    all_components.update(abstracted_robot.actuator_links.merged_components)
    print("Remaining leaves:",[link.name for link in locomotion_less_urdf.leaf_links])
    
    # ========== RECURSIVE BODY PART ABSTRACTION ======
    current_urdf = locomotion_less_urdf
    iteration = 0
    max_iterations = 5  # Protection against infinite loop
    
    bodypart_pipeline = AbstractComponentPipeline(keyword="body_part", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)

    while current_urdf is not None and len(current_urdf.leaf_links) > 0 and iteration < max_iterations:
        iteration += 1
        leaf_names = [link.name for link in current_urdf.leaf_links]
        print(f"\n================ [BODY PART PASS Iteration {iteration}] ================")
        print("ALL LINKS: ", [link.name for link in current_urdf.get_all_links()])
        print(f"Remaining leaf links to classify ({len(leaf_names)}): {leaf_names}")

        # Reset previous component containers
        abstracted_robot.body_part_links.components.clear()
        abstracted_robot.body_part_links.merged_components.clear()
        
        # Execute body part abstraction pass
        next_urdf = bodypart_pipeline.run_chain_pipeline(copy.deepcopy(current_urdf), abstracted_robot.body_part_links, full_verbose=True,
                                                         save_intermediate=False, load_intermediate=False)

        # Sync merged_components updated during this iteration
        all_components.update(abstracted_robot.body_part_links.merged_components)

        # Check for convergence (if run_chain_pipeline fails to reduce any leaf links)
        if next_urdf is None or len(next_urdf.leaf_links) == len(current_urdf.leaf_links):
            print(f"Warning: No leaf links were abstracted in iteration {iteration}. Halting loop.")
            break

        current_urdf = next_urdf

    print(f"\n Body part classification completed after {iteration} iteration(s).")
    if current_urdf is not None:
        print("ALL LINKS: ", [link.name for link in next_urdf.get_all_links()])
        print("Final remaining leaves:", [link.name for link in next_urdf.leaf_links])
        
    # bodypart_pipeline = AbstractComponentPipeline(keyword="body_part", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)
    # result_urdf = bodypart_pipeline.run_chain_pipeline(locomotion_less_urdf, abstracted_robot.body_part_links, full_verbose= True,
    #                                                             save_intermediate = False, 
    #                                                             load_intermediate = False)
    # all_components.update(abstracted_robot.body_part_links.components)
     
    print("Populating Interconnected Components:")
    bodypart_pipeline.resolve_and_assert_component_hierarchy(all_components, robot_name)
    
    ontology_handler.feeder.waitUpdate()
    ontology_handler.actions.save(EVAL_DIR + "/abstracted_" + robot_name + ".owl")
        

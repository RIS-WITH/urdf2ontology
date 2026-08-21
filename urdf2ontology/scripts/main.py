#!/usr/bin/env python3
import rclpy
import time
from urdf_tree import URDFTree
from AbstractedRobotLinks import AbstractedRobotLinks
from tasks.AbstractComponentPipeline import AbstractComponentPipeline

from tasks.ComponentLinkGroupingTask import *

from component_ontologies import *
import os

from ontopy import OntologyManipulator


if __name__ == '__main__':
    
    # DEFINES PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval"))
    ONTOLOGY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ontology"))
    URDF_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "urdf"))
        
    rclpy.init()
    ontology_handler = OntologyManipulator()
    
    # Load your OWL files
    files = ["component_ontology.owl"]
    for f in files:
        ontology_handler.actions.fadd(ONTOLOGY_DIR + "/" + f)
    ontology_handler.close()
    
    # URDF models to be evaluated
    eval_models = ['ingenuity.urdf', 'pepper.urdf','rbvogui_plus.urdf','pr2_standard.urdf', 'solo.urdf']

    # Select the index of the model to abstract
    robot_path_string = EVAL_DIR + "/" + eval_models[2]
    
    robot_name = robot_path_string.split('/')[-1].split('.')[0]

    urdf_tree = URDFTree(robot_name, robot_path_string)
    
    # Maintains a dict for each of the links in the URDF to keep track of which have been processed
    all_links = [link.name for link in urdf_tree.get_all_links()]
    print("init leaf links: ", [link.name for link in urdf_tree.leaf_links])
    
    # For full abstraction
    all_components = {}
    
    path_dict = {"main_path": SCRIPT_DIR, 
                 "eval": EVAL_DIR}
    
    abstracted_robot = AbstractedRobotLinks(robot_name, robot_path_string, urdf_tree)
    init_urdf = abstracted_robot.urdf_tree
    
    sensor_pipeline = AbstractComponentPipeline(keyword = "sensor", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)
    sensor_less_urdf = sensor_pipeline.run_tree_pipeline(init_urdf, abstracted_robot.sensor_links, verbose = True, 
                                                         save_intermediate = True, 
                                                         load_intermediate = False)
    all_components.update(abstracted_robot.sensor_links.merged_components)
    
    print("Remaining leaves:", [link.name for link in sensor_less_urdf.leaf_links])

    end_effector_pipeline = AbstractComponentPipeline(keyword = "end_effector", nb_runs = 1, ontology_handler = ontology_handler, paths = path_dict)
    end_effector_less_urdf = end_effector_pipeline.run_tree_pipeline(sensor_less_urdf, abstracted_robot.end_effector_links, verbose = True,
                                                                     save_intermediate = True, 
                                                                     load_intermediate = False)
    all_components.update(abstracted_robot.end_effector_links.merged_components)
    print("Remaining leaves:", [link.name for link in end_effector_less_urdf.leaf_links])

    locomotion_pipeline = AbstractComponentPipeline(keyword = "locomotion", nb_runs = 1, ontology_handler= ontology_handler, paths = path_dict)
    locomotion_less_urdf = locomotion_pipeline.run_tree_pipeline(end_effector_less_urdf, abstracted_robot.actuator_links, verbose = True,
                                                                save_intermediate = True, 
                                                                load_intermediate = False)
    all_components.update(abstracted_robot.actuator_links.merged_components)
    print("Remaining leaves:",[link.name for link in locomotion_less_urdf.leaf_links])
    
    bodypart_pipeline = AbstractComponentPipeline(keyword="body_part", nb_runs=1, ontology_handler = ontology_handler, paths = path_dict)
    result_urdf = bodypart_pipeline.run_chain_pipeline(locomotion_less_urdf, abstracted_robot.body_part_links, full_verbose= True,
                                                                save_intermediate = True, 
                                                                load_intermediate = False)
    all_components.update(abstracted_robot.body_part_links.components)

    if(result_urdf != None and len(result_urdf.leaf_links) >= 1):
        print("Remaining leaves:", [link.name for link in result_urdf.leaf_links])
     
    print("Populating Interconnected Components:")
    bodypart_pipeline.resolve_and_assert_component_hierarchy(all_components, robot_name)
    
    # Delay for ontologenius to process the latest modifications 
    time.sleep(2)
    
    ontology_handler.actions.save(EVAL_DIR + "/abstracted_" + robot_name + ".owl")
        

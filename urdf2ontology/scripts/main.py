#!/usr/bin/env python3
import rclpy

from urdf_tree import URDFTree
from AbstractedRobotLinks import AbstractedRobotLinks
from tasks.AbstractComponentPipeline import AbstractComponentPipeline

from tasks.ComponentLinkGroupingTask import *

from component_ontologies import *

# import sys
# sys.path.insert(0, "/home/bdussard/ros2_ws/src/ontologenius")

from ontopy import OntologyManipulator


if __name__ == '__main__':
    
    rclpy.init()
    ontology_handler = OntologyManipulator()
    # Load your OWL files
    files = [
        # "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/ontology/sensor_component.owl",
        # "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/ontology/end_effector_component.owl",
        # "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/ontology/structural_component.owl",
        "/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/ontology/combined_end_effector_sensor.owl",
    ]
    for f in files:
        ontology_handler.actions.fadd(f)
    ontology_handler.close()
    
    robot_models = ['/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/tiago.urdf',
                    '/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/pepper.urdf',
                    '/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/pr2.urdf',
                    '/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/baxter_2.urdf']

    robot_path_string = '/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/urdf/pr2.urdf'
    robot_name = robot_path_string.split('/')[-1].split('.')[0]

    urdf_tree = URDFTree(robot_path_string)
    # ================= ROBOT URDF ============

    abstracted_robot = AbstractedRobotLinks(robot_name, robot_path_string, urdf_tree)

    init_urdf = abstracted_robot.urdf_tree
    sensor_pipeline = AbstractComponentPipeline(keyword = "sensor", nb_runs = 3, 
                                                definitions_dict=definitions_dict['sensor'], 
                                                ontology=sensor_ontology, 
                                                ontology_handler= ontology_handler)
    sensor_less_urdf = sensor_pipeline.run_pipeline(init_urdf, abstracted_robot.sensor_links)
    
    end_effector_pipeline = AbstractComponentPipeline(keyword = "end_effector", nb_runs = 3,
                                                    definitions_dict=definitions_dict['end_effector'], 
                                                    ontology=end_effector_ontology,
                                                    ontology_handler= ontology_handler)
    end_effector_less_urdf = end_effector_pipeline.run_pipeline(sensor_less_urdf, abstracted_robot.end_effector_links)
    
    # Save the final populated ontology
    ontology_handler.actions.save("/home/bdussard/ros2_ws/src/urdf2ontology/urdf2ontology/results/asbtraction_" + robot_name + ".owl")
        
        

#!/usr/bin/env python3

from urdf_tree import URDFTree

from LLMInterface import LLMInterface

from tasks.SensorLinkIdentificationTask import SensorLinkIdentificationTask
from tasks.SensorLinkMountingPointTask import SensorLinkMountingPointTask
from tasks.SensorLinkGroupingTask import SensorLinkGroupingTask
from tasks.SensorLinkClassificationTask import SensorLinkClassificationTask
from tasks.SensorLinkAbstractingTask import SensorLinkAbstractingTask
from tasks.SensorLinkSplittingTask import SensorLinkSplittingTask

from AbstractedRobotLinks import AbstractedRobotLinks

# small_models = ["gemma3:4b", "mistral:7b", "llama3:8b"]
# average_models = ["gemma2:9b", "gemma3:12b", "mistral-nemo", "phi3:14b"]
# models_thinking = ["gpt-oss:20b", "qwen3:14b", "deepseek-r1:14b"]

# ========= DUMMY DATA ==========
sensor_subclasses = {"Bumper", "Sonar", "Camera", "Accelerometer", "Gyroscope", "TactileArea", "Laser", "IMU", "Microphone"}

camera_subclasses = {"ColorCamera", "RGBCamera", "DepthCamera", "InfraRedCamera", "StereoCamera", "MonochromeCamera"}

# ================= ROBOT URDF ============
robot_path = '../urdf/pr2.urdf'
urdf_tree = URDFTree(robot_path)
abstracted_robot = AbstractedRobotLinks("pr2", robot_path, urdf_tree)

robot_leaf_links = urdf_tree.get_leaf_links()
leaf_links_str = [leaf.name for leaf in robot_leaf_links]
robot_leaf_links_dict = {'links': leaf_links_str}
#frame_analyzer = URDFFrameAnalyzer(tree, None)

# ================= LLM MODULE ============
mistral = "mistral-nemo:12b"
gpt_thinking = "gpt-oss:20b"
deepseek_thinking = "deepseek-r1:14b"

llm_module = LLMInterface(model=gpt_thinking, temperature=0.7)

# ================= PIPELINE ============
# Get all the paths from a given node to its connected leaf nodes
# all_chains = urdf_tree.get_all_chain_names(urdf_tree.get_link('sensor_mount_link'))

# ========= IDENTIFICATION =======
# ==== Identify sensor links based on leaf names (restrict the subset of links)
sensor_identif = SensorLinkIdentificationTask(nb_runs=1)
result_identif = llm_module.run_task(sensor_identif, robot_leaf_links_dict, verbose=False)
print("<==== 1. IDENTIFIED SENSOR LINKS ====>")
print(result_identif)

abstracted_robot.add_identified_sensor_links(result_identif)

# ========= MOUNTING POINT =======
sensor_mounting = SensorLinkMountingPointTask(nb_runs=1)
result_mounting = llm_module.run_task(sensor_mounting, abstracted_robot.sensor_links.initial_chains, verbose=False)
print("<==== 2. IDENTIFIED SENSOR MOUNTING POINTS ====>")
print(result_mounting)

abstracted_robot.add_trimmed_sensor_chains(result_mounting)

# ========= GROUPING =======
sensor_grouping = SensorLinkGroupingTask(nb_runs=1)
result_grouping = llm_module.run_task(sensor_grouping, abstracted_robot.sensor_links.trimmed_chains, verbose=False)
print("<==== 3. GROUPING SENSOR LINKS ====>")
print(result_grouping)

abstracted_robot.add_sensor_groups(result_grouping)

# ========= CLASSIFICATION =======
sensor_classif = SensorLinkClassificationTask(sensor_subclasses, nb_runs=1)
result_classif = llm_module.run_task(sensor_classif, abstracted_robot.sensor_links.sensor_groups, verbose=False)
print("<==== 4. CLASSIFICATION SENSOR GROUPS ====>")
print(result_classif)

abstracted_robot.add_sensor_group_classification(result_classif)

# Gather the URDF chains for each sensor group, with children_links 

# ===== ABSTRACTING =========
# ==== Abstract each group based on the URDF frame (with children links to avoid removing sub-links)
# add for each for the identified group, all of the subchains from its lowest mounting_point
# output: only the ones that have been identified as only embedding links related to that sensor
# good example : (laser_tilt_mount_link + laser_tilt_link) -> laser_tilt individual connected to torso_lift_link
# bad example: (l_gripper_palm_link + l_gripper_motor_accelerometer_link) -> l_gripper_accelerometer because l_gripper_palm_link contains other links such as l_gripper_l_finger_tip_frame


# sensor_abstracting = SensorLinkAbstractingTask(nb_runs=1)
# result_abstracting = llm_module.run_task(sensor_abstracting, result_grouping, False)
# print("<==== 4. ABSTRACTING SENSOR GROUPS ====>", result_abstracting)

# # Fetch the sub-tree from each mounting_point
# sensor_sub_trees = []

# for sensor_group in result_abstracting.items():
#     sensor_id, mounting_point = sensor_group[0], sensor_group[1]
#     abstracted_robot.sensor_links.check_abstraction(sensor_id, mounting_point)
    


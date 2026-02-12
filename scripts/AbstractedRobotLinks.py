class AbstractedRobotLinks():
    def __init__(self, name, path, urdf_tree):
        self.robot_name = name
        self.urdf_path = path
        self.urdf_tree = urdf_tree
        self.sensor_links = SensorLinks()
        self.actuator_links = ActuatorLinks()
        self.end_effector_links = EndEffectorLinks()
        self.body_part_links = BodyPartLinks()
        
    def add_identified_sensor_links(self, result_identif):
        
        for identified_sensor in result_identif:
            chain_element = self.urdf_tree.get_tree_from_leaf(identified_sensor) #tree.get_tree_from_leaf(tree.get_link(identified_sensor))
            # Add the initial chain to the abstracted robot
            self.sensor_links.add_initial_chain(identified_sensor, chain_element)
    
    def add_trimmed_sensor_chains(self, result_mounting):
        
        # Store the trimmed subchain for each identified sensor, mounting point in the abstracted robot
        for link, mounting_point in result_mounting.items():
            involved_link_chain = self.sensor_links.initial_chains[link]
            mounting_point_index = involved_link_chain.index(mounting_point)
            trimmed_chain = involved_link_chain[mounting_point_index::]
            # store trimmed subchains
            self.sensor_links.add_trimmed_chain(link, trimmed_chain)
            
    def add_sensor_groups(self, result_grouping):
        # Create sensor groups
        for i in range(len(result_grouping)):
            sensor_key = "sensor_"+str(i)
            sensor_links, sensor_parent = result_grouping[i]

            sensor_subchains = []
            for link in sensor_links:
                index = self.sensor_links.trimmed_chains[link].index(sensor_parent)
                sensor_subchains.append(self.sensor_links.trimmed_chains[link][index::])

            self.sensor_links.add_sensor_group(sensor_key, sensor_links, sensor_subchains)
            
    def add_sensor_group_classification(self, result_classif):
        
        for sensor_id, sensor_type in result_classif.items():
            self.sensor_links.sensor_types[sensor_id] = sensor_type
        
class ActuatorLinks():
    def __init__(self):
        pass
    
class BodyPartLinks():
    def __init__(self):
        pass
    
class EndEffectorLinks():
    def __init__(self):
        pass

class SensorLinks():
    def __init__(self):
        self.initial_chains = {}
        self.trimmed_chains = {}
        # Contains actual sensor groups (sharing a similar mounting point)
        self.sensor_groups = {}
        self.mapping_link_groups = {}
        self.sensor_types = {}
        self.mapping_group_components = {}
        
    def add_initial_chain(self, link, chain):
        self.initial_chains[link] = chain
        
    def add_trimmed_chain(self, link, chain):
        self.trimmed_chains[link] = chain
        
    def add_sensor_group(self, sensor_id, links, chains):
        self.sensor_groups[sensor_id] = [links, chains]
        for link in links:
            self.mapping_link_groups[link]=sensor_id
            
    def add_sensor_classif(self, sensor_id, type):
        self.sensor_groups[sensor_id] = type
            
    def add_sensor_component(self, sensor_id, links, chains):
        self.sensor_groups[sensor_id] = [links, chains]
        for link in links:
            self.mapping_link_groups[link]=sensor_id
            
    def print_sensor_group(self, sensor_id):
        print("Sensor group: ", sensor_id)
        print("Links:", self.sensor_groups[sensor_id][0])
        print("Chains:", self.sensor_groups[sensor_id][1])
        
    def print_link(self, link_name):
        if(link_name in self.initial_chains.keys()):
            print("Identified Sensor Leaf with chain :", self.initial_chains[link_name])
        if(link_name in self.trimmed_chains.keys()):
            print("Split Sensor Leaf with chain :", self.trimmed_chains[link_name])
        if(link_name in self.mapping_link_groups.keys()):
            self.print_sensor_group(self.mapping_link_groups[link_name])
            
    # def check_abstraction(self, sensor_id: str, mounting_point : str) -> bool: 
    #     # ======= Returns True if none of the links in the subtree corresponding to the mounting_point have been forgotten ======
    #     print("Checking abstraction for ", sensor_id, 'mounted at', mounting_point)
    #     mounting_link = urdf_tree.get_link(mounting_point)
    #     sub_tree = urdf_tree.build_frame_context_from_link(mounting_link, [])
    #     #print("Sub tree is :", sub_tree)
        
    #     all_links_tree = urdf_tree.get_sublinks_from_link_str(urdf_tree.get_link(mounting_point))
    #     #print("all children links are :", all_links_tree)
        
    #     all_subchains = abstracted_robot.sensor_links.sensor_groups[sensor_id][1]
        
    #     identified_links_set = set()
    #     for subchain in all_subchains:
    #         identified_links_set.update(set(subchain))
    #     #print("All identified links are :", identified_links_set)
        
    #     if(len(all_links_tree) == 0):
    #         print("No children links")
        
    #     if(all_links_tree.issubset(identified_links_set) == True):
    #         print("All links are contained -> Abstraction is possible")
    #     else:
    #         print("Abstraction impossible because of the following links being forgotten:", list(all_links_tree - identified_links_set))
        
    
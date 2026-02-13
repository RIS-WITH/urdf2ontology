class ComponentLinks:
    def __init__(self):
        self.initial_chains = {}
        self.trimmed_chains = {}
        self.component_groups = {}
        self.mapping_link_groups = {}
        self.component_types = {}
        self.mapping_group_components = {}

    def add_initial_chain(self, link, chain):
        self.initial_chains[link] = chain

    def add_trimmed_chain(self, link, chain):
        self.trimmed_chains[link] = chain

    def add_component_group(self, component_id, links, chains):
        self.component_groups[component_id] = [links, chains]
        for link in links:
            self.mapping_link_groups[link] = component_id

    def add_component_classif(self, component_id, comp_type):
        self.component_types[component_id] = comp_type

    def print_component_group(self, component_id):
        print(f"Component group ({component_id}):")
        print("Links:", self.component_groups[component_id][0])
        print("Chains:", self.component_groups[component_id][1])

    def print_link(self, link_name):
        if link_name in self.initial_chains:
            print("Identified Leaf with chain:", self.initial_chains[link_name])
        if link_name in self.trimmed_chains:
            print("Trimmed Leaf chain:", self.trimmed_chains[link_name])
        if link_name in self.mapping_link_groups:
            self.print_component_group(self.mapping_link_groups[link_name])


class SensorLinks(ComponentLinks):
    def __init__(self):
        super().__init__()

class ActuatorLinks(ComponentLinks):
    def __init__(self):
        super().__init__()

class EndEffectorLinks(ComponentLinks):
    def __init__(self):
        super().__init__()

class BodyPartLinks(ComponentLinks):
    def __init__(self):
        super().__init__()

class AbstractedRobotLinks:
    def __init__(self, name, path, urdf_tree):
        self.robot_name = name
        self.urdf_path = path
        self.urdf_tree = urdf_tree
        self.sensor_links = SensorLinks()
        self.actuator_links = ActuatorLinks()
        self.end_effector_links = EndEffectorLinks()
        self.body_part_links = BodyPartLinks()

    def get_links_object(self, keyword):
        mapping = {
            "sensor": self.sensor_links,
            "actuator": self.actuator_links,
            "end_effector": self.end_effector_links,
            "body_part": self.body_part_links
        }
        return mapping[keyword]

    def add_identified_component_links(self, result_identif, keyword=""):
        links_obj = self.get_links_object(keyword)
        for identified_component in result_identif:
            chain_element = self.urdf_tree.get_tree_from_leaf(identified_component)
            links_obj.add_initial_chain(identified_component, chain_element)

    def add_trimmed_component_chains(self, result_mounting, keyword=""):
        links_obj = self.get_links_object(keyword)
        for link, mounting_point in result_mounting.items():
            involved_link_chain = links_obj.initial_chains[link]
            mounting_point_index = involved_link_chain.index(mounting_point)
            trimmed_chain = involved_link_chain[mounting_point_index:]
            links_obj.add_trimmed_chain(link, trimmed_chain)

    def add_component_groups(self, result_grouping, keyword=""):
        links_obj = self.get_links_object(keyword)
        for i, (component_links, component_parent) in enumerate(result_grouping):
            component_id = f"{keyword}_{i}"
            component_subchains = []
            for link in component_links:
                index = links_obj.trimmed_chains[link].index(component_parent)
                component_subchains.append(links_obj.trimmed_chains[link][index:])
            links_obj.add_component_group(component_id, component_links, component_subchains)

    def add_component_group_classification(self, result_classif, keyword=""):
        links_obj = self.get_links_object(keyword)
        for component_id, component_type in result_classif.items():
            links_obj.add_component_classif(component_id, component_type)


# class AbstractedRobotLinks():
#     def __init__(self, name, path, urdf_tree):
#         self.robot_name = name
#         self.urdf_path = path
#         self.urdf_tree = urdf_tree
#         self.sensor_links = SensorLinks()
#         self.actuator_links = ActuatorLinks()
#         self.end_effector_links = EndEffectorLinks()
#         self.body_part_links = BodyPartLinks()
        
#     def add_identified_component_links(self, result_identif):
        
#         for identified_component in result_identif:
#             chain_element = self.urdf_tree.get_tree_from_leaf(identified_component) #tree.get_tree_from_leaf(tree.get_link(identified_component))
#             # Add the initial chain to the abstracted robot
#             self.component_links.add_initial_chain(identified_component, chain_element)
    
#     def add_trimmed_component_chains(self, result_mounting):
        
#         # Store the trimmed subchain for each identified component, mounting point in the abstracted robot
#         for link, mounting_point in result_mounting.items():
#             involved_link_chain = self.component_links.initial_chains[link]
#             mounting_point_index = involved_link_chain.index(mounting_point)
#             trimmed_chain = involved_link_chain[mounting_point_index::]
#             # store trimmed subchains
#             self.component_links.add_trimmed_chain(link, trimmed_chain)
            
#     def add_component_groups(self, result_grouping):
#         # Create component groups
#         for i in range(len(result_grouping)):
#             component_key = "component_"+str(i)
#             component_links, component_parent = result_grouping[i]

#             component_subchains = []
#             for link in component_links:
#                 index = self.component_links.trimmed_chains[link].index(component_parent)
#                 component_subchains.append(self.component_links.trimmed_chains[link][index::])

#             self.component_links.add_component_group(component_key, component_links, component_subchains)
            
#     def add_component_group_classification(self, result_classif):
        
#         for component_id, component_type in result_classif.items():
#             self.component_links.component_types[component_id] = component_type
        
# class ActuatorLinks():
#     def __init__(self):
#         pass
    
# class BodyPartLinks():
#     def __init__(self):
#         pass
    
# class EndEffectorLinks():
#     def __init__(self):
#         pass

# class SensorLinks():
#     def __init__(self):
#         self.initial_chains = {}
#         self.trimmed_chains = {}
#         # Contains actual component groups (sharing a similar mounting point)
#         self.component_groups = {}
#         self.mapping_link_groups = {}
#         self.component_types = {}
#         self.mapping_group_components = {}
        
#     def add_initial_chain(self, link, chain):
#         self.initial_chains[link] = chain
        
#     def add_trimmed_chain(self, link, chain):
#         self.trimmed_chains[link] = chain
        
#     def add_component_group(self, component_id, links, chains):
#         self.component_groups[component_id] = [links, chains]
#         for link in links:
#             self.mapping_link_groups[link]=component_id
            
#     def add_component_classif(self, component_id, type):
#         self.component_groups[component_id] = type
            
#     def add_component_component(self, component_id, links, chains):
#         self.component_groups[component_id] = [links, chains]
#         for link in links:
#             self.mapping_link_groups[link]=component_id
            
#     def print_component_group(self, component_id):
#         print("Sensor group: ", component_id)
#         print("Links:", self.component_groups[component_id][0])
#         print("Chains:", self.component_groups[component_id][1])
        
#     def print_link(self, link_name):
#         if(link_name in self.initial_chains.keys()):
#             print("Identified Sensor Leaf with chain :", self.initial_chains[link_name])
#         if(link_name in self.trimmed_chains.keys()):
#             print("Split Sensor Leaf with chain :", self.trimmed_chains[link_name])
#         if(link_name in self.mapping_link_groups.keys()):
#             self.print_component_group(self.mapping_link_groups[link_name])
    
#     # def check_abstraction(self, component_id: str, mounting_point : str) -> bool: 
#     #     # ======= Returns True if none of the links in the subtree corresponding to the mounting_point have been forgotten ======
#     #     print("Checking abstraction for ", component_id, 'mounted at', mounting_point)
#     #     mounting_link = urdf_tree.get_link(mounting_point)
#     #     sub_tree = urdf_tree.build_frame_context_from_link(mounting_link, [])
#     #     #print("Sub tree is :", sub_tree)
        
#     #     all_links_tree = urdf_tree.get_sublinks_from_link_str(urdf_tree.get_link(mounting_point))
#     #     #print("all children links are :", all_links_tree)
        
#     #     all_subchains = abstracted_robot.component_links.component_groups[component_id][1]
        
#     #     identified_links_set = set()
#     #     for subchain in all_subchains:
#     #         identified_links_set.update(set(subchain))
#     #     #print("All identified links are :", identified_links_set)
        
#     #     if(len(all_links_tree) == 0):
#     #         print("No children links")
        
#     #     if(all_links_tree.issubset(identified_links_set) == True):
#     #         print("All links are contained -> Abstraction is possible")
#     #     else:
#     #         print("Abstraction impossible because of the following links being forgotten:", list(all_links_tree - identified_links_set))
        
    
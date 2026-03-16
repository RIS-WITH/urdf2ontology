from urdf_tree import URDFLink, URDFTree
from typing import List
from dataclasses import dataclass

@dataclass  
class ComponentGroup:
    component_id: str
    component_leaf_links: List[str]
    component_subchains: List[List[str]]
    component_type: int
    
class ComponentLinks():
    def __init__(self, urdf_tree : URDFTree):
        self.initial_urdf_tree = urdf_tree
        self.refined_urdf_tree = None
        self.grouped_components = {}
        
        self.initial_chains = {} # "link": subchain (identification task)
        self.trimmed_chains = {} # "link": [mount, trimmed_subchain] (mounting point task)
        self.component_preclassif = {} # "link": 'CameraFrame' (pre-classif task)
        self.component_postclassif = {} # "link": 'CameraFrame' (pre-classif task)
        self.component_groups = {} # {'sensor_0': [linkA, linkB, ...],} (grouping task)
        self.final_groups = {} # 'sensor_0': {[linkA, classifA, mountA, trimmedA], []} (grouping task)
        self.mapping_link_groups = {} # 'link": 'sensor_0'  (grouping task)
        self.component_types = {} # 'sensor_0': 'RGBCameraSystem' (post classification task)
        self.temp_components = {} # before verification (validation task)
        self.valid_components = {} # after verification (validation task)

    def add_initial_chain(self, link, chain):
        self.initial_chains.update({link: chain})
    
    def add_identified_component_links(self, result_identif):
        for identified_component in result_identif:
            chain_element = self.initial_urdf_tree.get_tree_from_leaf(identified_component)
            self.add_initial_chain(identified_component, chain_element)
            
    def add_trimmed_chain(self, link, mount, chain):
        self.trimmed_chains[link] = {'attachment_root': mount, 
                                     'kinematic_chain' : chain}
        
    def add_trimmed_component_chains(self, result_mounting):
        for link, mounting_point in result_mounting.items():
            involved_link_chain = self.initial_chains[link]
            mounting_point_index = involved_link_chain.index(mounting_point)
            trimmed_chain = involved_link_chain[mounting_point_index:]
            self.add_trimmed_chain(link, mounting_point, trimmed_chain)

    def add_preclassif(self, result):
        for link, concept in result.items():
            self.component_preclassif.update({link: concept})
            
    def add_postclassif(self, result):
        for group_id, concept in result.items():
            self.component_postclassif.update({group_id: concept})
        
    def add_component_group(self, component_id, links, chains):
        self.component_groups[component_id] = [links, chains]
        for link in links:
            self.mapping_link_groups[link] = component_id

    def add_component_groups(self, result_grouping, keyword):
        for i, component_links in enumerate(result_grouping):
            component_id = f"{keyword}_{i}"
            component_subchains = []

            for link in component_links:
                component_subchains.append(link)
            self.component_groups[component_id] = component_subchains # we store the list of links in the group ["linkA", "linkB"]
            

    def add_component_classif(self, component_id, comp_type):
        self.component_types[component_id] = comp_type
        
    def add_component_group_classification(self, result_classif):
        for component_id, component_type in result_classif.items():
            self.add_component_classif(component_id, component_type)
            
    def create_component_group(self, verbose = False):

        for component_id in self.component_groups.keys():
            if(self.component_types[component_id] != None and self.component_groups[component_id] != None):
                
                new_component = ComponentGroup(component_id=component_id, 
                                               component_leaf_links=self.component_groups[component_id][0], 
                                               component_subchains=self.component_groups[component_id][1], 
                                               component_type=self.component_types[component_id])
                
                self.temp_components[component_id] = new_component
                if(verbose):
                    self.print_component_group(component_id)
    
    def abstract_component(self, keyword = "", verbose = False) -> URDFTree:
        print(f"Abstracting {keyword} components")
        # create a deep copy of the initial tree
        refined_urdf_tree = self.initial_urdf_tree.deepcopy()
      
        # modify the refined_tree to abstract away valid components
        for component_id, valid_component in self.valid_components.items():
            new_link_name = valid_component.component_type.lower() + "_" + valid_component.component_id
            refined_urdf_tree.replace_subtree(valid_component.component_mounting_link, URDFLink(new_link_name))
            # append to the sensor_links
            refined_urdf_tree.sensor_links.append(refined_urdf_tree.get_link(new_link_name))
        print(refined_urdf_tree.sensor_links)
        return refined_urdf_tree
            
    def print_component_group(self, component_id):
        selected_component = self.temp_components[component_id]
        if(selected_component == None):
            print("Invalid component identifier")
        else:
            print("Component: ", component_id)
            print("Component Type:", selected_component.component_type)
            # print("Mounting Link:", selected_component.component_mounting_link)
            print("Leaf Links:", selected_component.component_leaf_links)
            print("Subchains:")
            for chain in selected_component.component_subchains:
                print(chain)

    def print_link(self, link_name):
        if link_name in self.initial_chains:
            print("Identified Leaf with chain:", self.initial_chains[link_name])
        if link_name in self.trimmed_chains:
            print("Trimmed Leaf chain:", self.trimmed_chains[link_name])
        if link_name in self.mapping_link_groups:
            self.print_component_group(self.mapping_link_groups[link_name])
    
    def build_component_groups(self, trimmed_chains, component_preclassif, grouped_links):
        """
        Args:
            trimmed_chains: {"link": [mount, trimmed_subchain]} - From Mounting Task
            component_preclassif: {"link": "Type"} - From Pre-classification Task
            grouped_links: {'sensor_0': [linkA, linkB, ...]} - From Grouping Task
            
        Returns:
            Dict: { 'sensor_0': [(link, type, mount, trimmed_subchain), ...] }
        """
        final_component_groups = {}
        
        # Iterate over the existing groups defined by the Grouping Task
        for group_id, links in grouped_links.items():
            group_tuples = []
            
            for link in links:
                # 1. Retrieve the pre-classified type for this link
                link_type = component_preclassif.get(link, f"UnknownSensor") # needs to be set with the CamelCase keyword version
                
                # 2. Retrieve mounting and subchain info from trimmed_chains
                # mount_info is expected to be [mount, trimmed_subchain]
                mount_info = trimmed_chains.get(link)
                
                if mount_info:
                    mount = mount_info['attachment_root']
                    trimmed_subchain = mount_info['kinematic_chain']
                else:
                    # Fallback if the link wasn't processed in the mounting task
                    mount = "UnknownMount"
                    trimmed_subchain = "NoSubchain"
                
                # 3. Append the tuple to the current group's list
                group_tuples.append((link, link_type, mount, trimmed_subchain))
                
            # Assign the list of tuples to the group ID
            final_component_groups[group_id] = group_tuples
        # append to final group
        self.final_groups = final_component_groups
        # return final_component_groups
        
    def build_final_group_mapping(self, final_component_groups, group_states):
        """
        Args:
            final_component_groups: { 'sensor_0': [(link, subpart, mount, subchain), ...] }
            group_states: { 'sensor_0': 'RGBD_Camera' }
            
        Returns:
            Dict: { 
                'sensor_0': {
                    'component_type': 'RGBD_Camera',
                    'links': ['kinect_ir_link', 'kinect_rgb_link'],
                    'details': {
                        'kinect_ir_link': {'subpart': 'IrReceiver', 'mount': 'head', 'subchain' : []},
                        'kinect_rgb_link': {'subpart': 'RgbFrame', 'mount': 'head', 'subchain': []}
                    }
                }
            }
        """
        final_output = {}

        for group_id, component_type in group_states.items():
            # Initialize the group entry
            final_output[group_id] = {
                "component_type": component_type,
                "links": [],
                "details": {}
            }
            
            # Pull link-specific data from the intermediate tuple list
            tuple_list = final_component_groups.get(group_id, [])
            
            for link, subpart, mount, subchain in tuple_list:
                # 1. Add to the flat list of links
                final_output[group_id]["links"].append(link)
                
                # 2. Add specific metadata for that link
                final_output[group_id]["details"][link] = {
                    "subpart_type": subpart,
                    "mount": mount,
                    "subchain": subchain
                }
                
        return final_output
            
        
class SensorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
    
    def verify_connectivity(self, identified_links_set) -> bool:

        visited = set()
        stack = [next(iter(identified_links_set))]

        while stack:
            current_name = stack.pop()

            if current_name not in visited:
                visited.add(current_name)

                current_link = self.initial_urdf_tree.get_link(current_name)
                neighbors = self.initial_urdf_tree.get_neighbors(current_link)

                for neighbor in neighbors:
                    neighbor_name = neighbor.name
                    if neighbor_name in identified_links_set and neighbor_name not in visited:
                        stack.append(neighbor_name)

        if visited != identified_links_set:
            print("Sensor group is not connected")
            print("Visited:", visited)
            print("Expected:", identified_links_set)
            return False

        print("Connectivity check passed")
        return True
        
    def verify_internal_completeness(self, sensor, identified_links_set) -> bool:

        mounting_link = sensor.component_mounting_link

        for link_name in identified_links_set:
            link = self.initial_urdf_tree.get_link(link_name)

            for child in link.children:
                # If child is outside group
                if child.name not in identified_links_set:

                    # Allowed only if this is the mounting link
                    if link_name == mounting_link:
                        continue

                    print("Missing internal descendant:", child.name)
                    return False

        print("Internal completeness check passed")
        return True
    
    def verify_single_attachment(self, identified_links_set) -> bool:
        external_neighbors = set()

        for link_name in identified_links_set:
            urdf_link = self.initial_urdf_tree.get_link(link_name)
            neighbors = self.initial_urdf_tree.get_neighbors(urdf_link)

            for neighbor in neighbors:
                neighbor_name = neighbor.name
                if neighbor_name not in identified_links_set:
                    external_neighbors.add(neighbor_name)

        if len(external_neighbors) != 1:
            print("Invalid number of external attachments:", external_neighbors)
            return False

        mounting_link = list(external_neighbors)[0]
        print("Single external attachment check passed")
        print("-> Mounting link detected as:", mounting_link)
        return True
         
    def verify_abstraction(self) -> List[ComponentGroup]:
        # each sensor group is (id, class, mounting_point, leaf_links, subchains)
        # if verified, create hasMountingLink/isMountingLinkOf, hasInternalLink (abstracted links), hasInternalJoint, observesPhysicalQuantity some Image, Depth, etc
        # A connected, internally closed subgraph with exactly one external interface.
        print("Verifying sensors ==== ")
        validated_components, rejected_components = [], []

        for sensor_id in self.temp_components:
            print(sensor_id)
            self.print_component_group(sensor_id)
            sensor = self.temp_components[sensor_id]
            print("Verifying sensor:", sensor.component_id)
            
            identified_links_set = set()
            for subchain in sensor.component_subchains:
                identified_links_set.update(set(subchain))
                
            if not identified_links_set:
                print("Sensor group is empty")
                rejected_components.append(sensor)
                continue
            
            # checking that each of the branch in the subchains is indeed connected
            if(self.verify_connectivity(identified_links_set) == False):
                rejected_components.append(sensor)
                continue
            
            if(self.verify_internal_completeness(sensor, identified_links_set) == False):
                rejected_components.append(sensor)
                continue
            
            # verify that each of the leaf links in the group attach at a single mounting point
            if(self.verify_single_attachment(identified_links_set) == False):
                rejected_components.append(sensor) # if false -> THEN ONLY ABSRACT THE IDENTIFIED LEAF LINK or if there is several chains the component, common ancestor
                continue
            
            validated_components.append(sensor)
            
        return (validated_components, rejected_components)

class ActuatorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
        
    def verify_abstraction(self) -> List[ComponentGroup]:
        print("Verifiying")

class EndEffectorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
        
    def verify_abstraction(self) -> List[ComponentGroup]:
        print("Verifiing")
        
class BodyPartLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
        
    def verify_abstraction(self) -> List[ComponentGroup]:
        print("Verifiing")

class AbstractedRobotLinks:
    def __init__(self, name, path, urdf_tree):
        self.robot_name = name
        self.urdf_path = path
        self.urdf_tree = urdf_tree
        self.sensor_links = SensorLinks(urdf_tree)
        self.actuator_links = ActuatorLinks(urdf_tree)
        self.end_effector_links = EndEffectorLinks(urdf_tree)
        self.body_part_links = BodyPartLinks(urdf_tree)

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
            links_obj.add_trimmed_chain(link, mounting_point, trimmed_chain)

    def add_component_groups(self, result_grouping, keyword=""):
        links_obj = self.get_links_object(keyword)
        for i, (component_links, component_parent) in enumerate(result_grouping):
            component_id = f"{keyword}_{i}"
            component_subchains = []
            for link in component_links:
                index = links_obj.trimmed_chains[link][1].index(component_parent)
                component_subchains.append(links_obj.trimmed_chains[link][1][index:])
            links_obj.add_component_group(component_id, component_parent, component_links, component_subchains)

    def add_component_group_classification(self, result_classif, keyword=""):
        links_obj = self.get_links_object(keyword)
        for component_id, component_type in result_classif.items():
            links_obj.add_component_classif(component_id, component_type)
            
    def create_component_groups(self, keyword ="", verbose = False):
        # Create Component group
        links_obj = self.get_links_object(keyword)
        links_obj.create_component_group(verbose)
        print("--Created", len(links_obj.temp_components), keyword + " components")
        # Verify Component group
        validated_components, rejected_components = links_obj.verify_sensor_abstraction()
        links_obj.valid_components = validated_components
        for component in validated_components:
            links_obj.valid_components[component.component_id] = component
        print("--Validated", len(validated_components), keyword + " components")
        print("--Rejected", len(rejected_components), keyword + " components") # need to be manually verified by an expert
        
    def abstract_component(self, keyword = "", verbose = False) -> URDFTree:
        print(f"Abstracting {keyword} components")
        links_obj = self.get_links_object(keyword)
        for valid_component in links_obj.valid_components:
            # return the abstracted self.refined_urdf_tree
            print("abstracting")
        
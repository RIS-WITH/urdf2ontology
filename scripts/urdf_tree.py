import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional

class URDFLink:
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional['URDFLink'] = None
        self.children: List['URDFLink'] = []
        self.joint_to_parent_type: Optional[str] = None
        self.joint_name: Optional[str] = None
        self.visual_count: int = 0
        self.collision_count: int = 0
        self.gazebo_elem = None
        
    def add_child(self, child: 'URDFLink'):
        """Add a child link to this link"""
        child.parent = self
        self.children.append(child)
        
    def __repr__(self):
        return f"Link(name='{self.name}', parent='{self.parent.name if self.parent else None}', " \
               f"children={[c.name for c in self.children]}, joint='{self.joint_name}', " \
               f"joint_type='{self.joint_to_parent_type}', visuals={self.visual_count}, " \
               f"collisions={self.collision_count})"

class URDFTree:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.links: Dict[str, URDFLink] = {}
        self.root: Optional[URDFLink] = None
        self.leaf_links: List[URDFLink] = []
        self.sensor_links: List[URDFLink] = []
        self.actuator_links: List[URDFLink] = []
        self._parse_urdf()
        
    def _parse_urdf(self):
        """Parse the URDF file and build the tree structure"""
        tree = ET.parse(self.urdf_path)
        root_element = tree.getroot()
        
        # Parse all links first
        for link_element in root_element.findall('link'):
            link_name = link_element.get('name')
            if link_name:
                link = URDFLink(link_name)
                self.links[link_name] = link
                
                # Count visuals and collisions
                link.visual_count = len(link_element.findall('visual'))
                link.collision_count = len(link_element.findall('collision'))
        
        # Parse all joints and build connections
        for joint_element in root_element.findall('joint'):
            joint_name = joint_element.get('name')
            joint_type = joint_element.get('type')
            
            # Get parent and child links
            parent_link = joint_element.find('parent')
            child_link = joint_element.find('child')
            
            if parent_link is not None and child_link is not None:
                parent_name = parent_link.get('link')
                child_name = child_link.get('link')
                
                if parent_name in self.links and child_name in self.links:
                    parent = self.links[parent_name]
                    child = self.links[child_name]
                    
                    # Set joint info on child link
                    child.joint_name = joint_name
                    child.joint_to_parent_type = joint_type
                    
                    # Build tree structure (parent adds child)
                    parent.add_child(child)
        
        # Maybe not necessary to check that the sensor type is valid (if we assume that the URDF file is correct)
        valid_sensor_types = ["air_pressure", "air_speed", "altimeter", "camera", "contact", "depth",
                              "depth_camera", "force_torque", "gps", "imu", "lidar", "logical_camera", 
                              "magnetometer", "navsat", "ray", "rfidtag", "sonar", "transceiver"]
        
        # Parse all gazebo elements and complement the corresponding links
        for gazebo_element in root_element.findall('gazebo'):
            reference_link = gazebo_element.get('reference')
            for sensor_element in gazebo_element.findall('sensor'):
                sensor_type = sensor_element.get('type')
    
                # Store the type of sensor found, but does not process sensor parameters (like camera width for instance)
                # If feature desired, then check out the SDF webpage (https://sdformat.org/spec/1.12/sensor/) to resolve which parameters
                if(sensor_type in valid_sensor_types):
                    self.links[reference_link].gazebo_elem = "sensor_" + sensor_type
                    print("Found valid sensor type: ", sensor_type, " on reference link ", reference_link)
                    self.sensor_links.append(self.links[reference_link])
                else:
                    print("Unsupported sensor type: ", sensor_type, " on reference link ", reference_link)
                
        # Find root and leaf nodes
        self._find_root_and_leaves()
        
    def _find_root_and_leaves(self):
        """Identify root and leaf links in the tree"""
        # Find all parent and child links
        all_parents = set()
        all_children = set()
        
        # Collect all parent-child relationships
        for link in self.links.values():
            if link.parent:
                all_parents.add(link.parent.name)
            for child in link.children:
                all_children.add(child.name)
        
        # Root links have no parent (are in all_links but not in all_children)
        root_names = set(self.links.keys()) - all_children
        if root_names:
            self.root = self.links[list(root_names)[0]]  # Pick first root
            
        # Leaf links have no children (are in all_links but not in all_parents)
        self.leaf_links = [link for link in self.links.values() if not link.children]
        
    def get_link(self, name: str) -> Optional[URDFLink]:
        """Get a link by name"""
        return self.links.get(name)
        
    def get_all_links(self) -> List[URDFLink]:
        """Get all links in the tree"""
        return list(self.links.values())
        
    def get_root(self) -> Optional[URDFLink]:
        """Get the root link"""
        return self.root
        
    def get_leaf_links(self) -> List[URDFLink]:
        """Get all leaf links"""
        return self.leaf_links
        
    def get_tree_from_root(self) -> List[URDFLink]:
        """Get tree structure starting from root (breadth-first traversal)"""
        if not self.root:
            return []
            
        result = []
        queue = [self.root]
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            # Add children to queue for processing
            queue.extend(current.children)
            
        return result
    
    def get_subtree_from_link(self, link_name):
        """Get subtree structure starting from link_name """
        tree_cpy = self
        
        # The provided link_name does not exist in the tree
        if link_name not in self.links.keys():
            return []
            
        result = []
        queue = [self.links[link_name]]
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            # Add children to queue for processing
            queue.extend(current.children)
            
        return result
    
    def get_sublinks_from_link_str(self, link_name) -> set:
        """Get subtree structure starting from link_name """
        tree_cpy = self
        result = set()
        
        # The provided link_name does not exist in the tree
        if link_name not in self.links.keys():
            return result
    
        queue = [self.links[link_name]]
        
        while queue:
            current = queue.pop(0)
            result.update(set(current.name))
            # Add children to queue for processing
            queue.extend(current.children)
            
        return result
        
    def get_tree_from_leaf(self, leaf: URDFLink) -> List[URDFLink]:
        """Get tree structure from leaf to root (path up to root)"""
        if not leaf:
            return []
            
        path = []
        current = leaf
        while current:
            path.append(current)
            current = current.parent
            
        # Return path from root to leaf
        return path[::-1]  # Reverse to get root to leaf path
    
    def get_tree_from_leaf(self, leaf: str) -> List[str]:
        """Get tree structure from leaf to root (path up to root)"""
        leaf_link = self.get_link(leaf)
        if not leaf_link:
            return []
            
        path = []
        current = leaf_link
        while current:
            path.append(current.name)
            current = current.parent
            
        # Return path from root to leaf
        return path[::-1]  # Reverse to get root to leaf path

    def get_branches_from_node(self, node: str) -> List[List[str]]:
        
        link = self.get_link(node)
        if(link == None):
            print("Link" + node + "does not exist in the tree")
        else:
            branches = []
            
    def get_all_chain_names(self, start: URDFLink) -> List[List[str]]:
        """
        Returns all chains from the given link to leaf links.
        Each chain is a list of link names.
        """
        chains = []

        def dfs(node: URDFLink, path: List[str]):
            path.append(node.name)

            if not node.children:
                chains.append(path.copy())
            else:
                for child in node.children:
                    dfs(child, path)

            path.pop()

        dfs(start, [])
        return chains

    def print_tree(self, max_depth: int = 10):
        """Print the tree structure"""
        if not self.root:
            print("No root found")
            return
            
        def print_node(node, depth=0):
            if depth >= max_depth:
                return
            indent = "  " * depth
            print(f"{indent}{node.name} (joint: {node.joint_name}, type: {node.joint_to_parent_type})")
            for child in node.children:
                print_node(child, depth + 1)
                
        print_node(self.root)

    def export_urdf(self, output_path: str, robot_name="robot"):
        
        robot = ET.Element(robot_name)
        robot.set("name", robot_name)
        
        # Create the links with no visual, collision or geomtery
        for link in self.links.values():
            link_elem = ET.SubElement(robot, "link")
            link_elem.set("name", link.name)

        # Create the joints between links with parent, children, and joint type (with dummy values for revolute and prismatic)
        for link in self.links.values():
            if link.parent is None:
                continue
            
            joint_elem = ET.SubElement(robot, "joint")
            joint_elem.set("name", link.joint_name or f"{link.parent.name}_to_{link.name}")
            joint_elem.set("type", link.joint_to_parent_type or "fixed")

            # Dummy values
            if link.joint_to_parent_type in ("revolute", "prismatic"):
                ET.SubElement(joint_elem,"limit",lower="0.0", upper="0.0", effort="0.0",velocity="0.0",)
                
            ET.SubElement(joint_elem, "parent", link=link.parent.name)
            ET.SubElement(joint_elem, "child", link=link.name)

        xml_str = ET.tostring(robot, "utf-8")
        pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")

        with open(output_path, "w") as f:
            f.write(pretty)

    def build_frame_context(self, link: URDFLink) -> Dict:
        """Build comprehensive context about the frame for analysis"""
        return {
            'frame_name': link.name,
            'parent_link': link.parent.name if link.parent else None,
            'children_links': [child.name for child in link.children],
            'joint_name': link.joint_name,
            'joint_type': link.joint_to_parent_type,
            'parent_joint_type': link.parent.joint_to_parent_type if link.parent else None,
            'parent_joint_name': link.parent.joint_name if link.parent else None,
        }
        
    def build_frame_context_without_joints(self, link: URDFLink) -> Dict:
        """Build comprehensive context about the frame for analysis"""
        return {
            'frame_name': link.name,
            'children_links': [child.name for child in link.children],
            'joint_type': link.joint_to_parent_type,
        }
    
    def build_frame_context_from_link(self, link: URDFLink, frames: List) -> List:
        """Build comprehensive context from the initial frame and its children for analysis"""
        
        if frames is None:
            frames = []

        if link is None:
            return frames

        frames.append(self.build_frame_context_without_joints(link))

        for child in link.children or []:
            self.build_frame_context_from_link(child, frames)

        return frames
  
        
if __name__ == "__main__":
    # For this test, let's create a proper, clean output
    mock_urdf_content = """<?xml version="1.0"?>
    <robot name="test_robot">
    <link name="base_link">
        <visual>
        <geometry>
            <box size="1 1 1"/>
        </geometry>
        </visual>
        <collision>
        <geometry>
            <box size="1 1 1"/>
        </geometry>
        </collision>
    </link>
    
    <link name="shoulder_link">
        <visual>
        <geometry>
            <cylinder length="1" radius="0.5"/>
        </geometry>
        </visual>
    </link>
    
    <link name="elbow_link">
        <collision>
        <geometry>
            <sphere radius="0.3"/>
        </geometry>
        </collision>
    </link>
    
    <link name="wrist_link"/>
    
    <joint name="shoulder_joint" type="revolute">
        <parent link="base_link"/>
        <child link="shoulder_link"/>
        <axis xyz="0 0 1"/>
    </joint>
    
    <joint name="elbow_joint" type="revolute">
        <parent link="shoulder_link"/>
        <child link="elbow_link"/>
        <axis xyz="0 1 0"/>
    </joint>
    
    <joint name="wrist_joint" type="revolute">
        <parent link="elbow_link"/>
        <child link="wrist_link"/>
        <axis xyz="1 0 0"/>
    </joint>
    </robot>"""

    # Write mock URDF to file for testing
    with open('test_robot.urdf', 'w') as f:
        f.write(mock_urdf_content)

    # Parse the URDF
    tree = URDFTree('/home/bdussard/urdf2ontology/urdf/pr2.urdf')

    # Let's simulate what the correct output should be based on your expected structure:
    print("=== CORRECTED OUTPUT ===")
    print("=== Tree Structure ===")
    tree.print_tree()

    print("\n=== Root Link ===")
    if tree.get_root():
        print(tree.get_root())
        
    print("\n=== Leaf Links ===")
    for leaf in tree.get_leaf_links():
        print(leaf.name)

    print("\n=== Tree from Root ===")
    from_root = tree.get_tree_from_root()
    for link in from_root:
        print(f"  {link.name}")

    print("\n=== Tree from Leaf (wrist_link) ===")
    wrist_link = tree.get_link("wrist_link")
    if wrist_link:
        path = tree.get_tree_from_leaf(wrist_link)
        for link in path:
            print(f"  {link.name}")

    print("\n=== All Links ===")
    for link in tree.get_all_links():
        print(f"  {link}")

    print("\n=== Verification of relationships ===")
    print("base_link children:", [c.name for c in tree.get_link("base_link").children])
    print("l_shoulder_link children:", [c.name for c in tree.get_link("l_shoulder_pan_link").children])
    print("elbow_link children:", [c.name for c in tree.get_link("elbow_link").children])
    print("wrist_link children:", [c.name for c in tree.get_link("wrist_link").children])

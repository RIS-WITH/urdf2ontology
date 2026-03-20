import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional, Set

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
        
    def clone_shallow(self) -> "URDFLink":
        new_link = URDFLink(self.name)
        new_link.joint_to_parent_type = self.joint_to_parent_type
        new_link.joint_name = self.joint_name
        new_link.visual_count = self.visual_count
        new_link.collision_count = self.collision_count
        new_link.gazebo_elem = self.gazebo_elem
        return new_link
        
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
        self.end_effector_links: List[URDFLink] = []
        self.body_part_links: List[URDFLink] = []
        self._parse_urdf()

    def abstract_components(self, final_groups: Dict, keep_abstracted_links = False):
        """
        Collapses multi-link components into single semantic URDF links.
        Handles parallel attachments, chained sub-trees, and cleans up the tree dictionary.
        """
        for gid, group_data in final_groups.items():
            comp_type = group_data['component_type']
            leaf_names = group_data['links']
            
            # Get the mount point (e.g., 'head') from the first link's metadata
            first_leaf = leaf_names[0]
            mount_name = group_data['details'][first_leaf]['mount']
            
            if mount_name not in self.links:
                continue
                
            mount_link = self.links[mount_name]

            # 1. Identify "Entry Points": Links in the group attached directly to the mount
            # This handles cases where a device has multiple parallel roots
            roots_to_prune = set()
            for name in leaf_names:
                if name not in self.links: continue
                curr = self.links[name]
                # Traverse up until we find the link that connects to the mount
                while curr.parent:
                    if curr.parent.name == mount_name:
                        roots_to_prune.add(curr)
                        break
                    curr = curr.parent

            if not roots_to_prune:
                continue

            # 2. Create the New Semantic Link (e.g., 'RGBD_Camera_0')
            new_link_name = f"{comp_type}_{gid.split('_')[-1]}"
            new_component = URDFLink(new_link_name)
            
            # Inherit joint properties from the first entry point found
            example_root = list(roots_to_prune)[0]
            new_component.joint_to_parent_type = example_root.joint_to_parent_type
            new_component.joint_name = f"joint_{new_link_name}_to_{mount_name}"

            # 3. Recursive Purge: Remove all sub-tree links from self.links dictionary
            for root_link in roots_to_prune:
                self._recursive_purge(root_link)

            # 4. Prune from Tree Structure: Remove the old roots from the mount's children
            root_names = [r.name for r in roots_to_prune]
            mount_link.children = [c for c in mount_link.children if c.name not in root_names]

            if(keep_abstracted_links == True):
                # 5. Inject the Abstraction
                mount_link.add_child(new_component)
                self.links[new_link_name] = new_component
            
                # 6. Final Sync
                # Remove deleted links and add the new abstracted leaf
                self.leaf_links.append(new_component)
            self.refresh_leaf_links()

    def _recursive_purge(self, link: URDFLink):
        """Removes a link and all its descendants from the tree's internal dictionary."""
        # Recursively handle all children first
        for child in list(link.children):
            self._recursive_purge(child)
        
        # Remove this link from the global dictionary
        if link.name in self.links:
            del self.links[link.name]
            
    def refresh_leaf_links(self):
        """
        Re-filters the leaf_links list to only include links that 
        still exist in the 'self.links' dictionary.
        """
        self.leaf_links = [l for l in self.leaf_links if l.name in self.links]
    
    def deepcopy(self) -> "URDFTree":
        """
        Create a deep copy of this URDFTree, including all links
        and their parent/child relationships.
        """
        # Create empty tree instance without parsing again
        new_tree = URDFTree.__new__(URDFTree)

        # Copy simple attributes
        new_tree.urdf_path = self.urdf_path
        new_tree.links = {}
        new_tree.root = None
        new_tree.leaf_links = []
        new_tree.sensor_links = []
        new_tree.actuator_links = []

        # --- Step 1: Create all link copies (without relationships) ---
        for name, link in self.links.items():
            new_tree.links[name] = link.clone_shallow()
            # new_link = URDFLink(name)
            # new_link.joint_to_parent_type = link.joint_to_parent_type
            # new_link.joint_name = link.joint_name
            # new_link.visual_count = link.visual_count
            # new_link.collision_count = link.collision_count
            # new_link.gazebo_elem = link.gazebo_elem  # copy reference (deepcopy if needed)

            # new_tree.links[name] = new_link

        # --- Step 2: Reconstruct hierarchy ---
        for name, link in self.links.items():
            new_link = new_tree.links[name]

            if link.parent:
                parent_copy = new_tree.links[link.parent.name]
                parent_copy.add_child(new_link)

        # --- Step 3: Copy root ---
        if self.root:
            new_tree.root = new_tree.links[self.root.name]

        # --- Step 4: Rebuild leaf list ---
        for link in new_tree.links.values():
            if not link.children:
                new_tree.leaf_links.append(link)

        # --- Step 5: Rebuild sensor/actuator lists ---
        new_tree.sensor_links = [new_tree.links[link.name] for link in self.sensor_links]
        new_tree.actuator_links = [new_tree.links[link.name] for link in self.actuator_links]

        return new_tree
    
    def _collect_descendants(self, root: URDFLink) -> Set[URDFLink]:
        """Return all descendants of root INCLUDING root."""
        visited = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(node.children)

        return visited

    def _collect_leaves(self, nodes: Set[URDFLink]) -> Set[URDFLink]:
        """Return leaves within a given node set."""
        leaves = set()
        for node in nodes:
            if not any(child in nodes for child in node.children):
                leaves.add(node)
        return leaves

    def _minimal_subtree(self, mounting_point: URDFLink, sensor_leaves: Set[URDFLink]) -> Set[URDFLink]:
        """
        Return all nodes on paths from mounting_point
        to each sensor leaf.
        """
        subtree_nodes = set()

        for leaf in sensor_leaves:
            current = leaf
            while current is not None and current != mounting_point:
                subtree_nodes.add(current)
                current = current.parent

            if current != mounting_point:
                raise ValueError(f"Leaf '{leaf.name}' is not a descendant of mounting point '{mounting_point.name}'")

        subtree_nodes.add(mounting_point)
        return subtree_nodes
    
    def replace_subtree(self, subtree_root_name: str, replacement_link: URDFLink):
        """
        Replace the subtree rooted at 'subtree_root_name' with a single 'replacement_link'.

        The replacement link will take the place of the subtree root in the original tree hierarchy.
        """
        if subtree_root_name not in self.links:
            raise ValueError(f"Link '{subtree_root_name}' not found in tree.")

        old_root = self.links[subtree_root_name]
        parent = old_root.parent

        # --- Step 1: Collect all nodes in subtree ---
        nodes_to_remove = []

        def collect(node: URDFLink):
            nodes_to_remove.append(node)
            for child in node.children:
                collect(child)

        collect(old_root)

        # --- Step 2: Detach from parent ---
        if parent:
            parent.children = [c for c in parent.children if c != old_root]
        else:
            # replacing actual root
            self.root = replacement_link

        # --- Step 3: Remove nodes from registry ---
        for node in nodes_to_remove:
            self.links.pop(node.name, None)

        # --- Step 4: Insert replacement ---
        replacement_link.parent = parent
        replacement_link.children = []

        if parent:
            parent.add_child(replacement_link)

        self.links[replacement_link.name] = replacement_link

        for link in self.links.values():
            if not link.children:
                self.leaf_links.append(link)
    
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
        
        # # Parse all gazebo elements and complement the corresponding links
        # for gazebo_element in root_element.findall('gazebo'):
        #     reference_link = gazebo_element.get('reference')
        #     for sensor_element in gazebo_element.findall('sensor'):
        #         sensor_type = sensor_element.get('type')
    
        #         # Store the type of sensor found, but does not process sensor parameters (like camera width for instance)
        #         # If feature desired, then check out the SDF webpage (https://sdformat.org/spec/1.12/sensor/) to resolve which parameters
        #         if(sensor_type in valid_sensor_types):
        #             self.links[reference_link].gazebo_elem = "sensor_" + sensor_type
        #             print("Found valid sensor type: ", sensor_type, " on reference link ", reference_link)
        #             self.sensor_links.append(self.links[reference_link])
        #         else:
        #             print("Unsupported sensor type: ", sensor_type, " on reference link ", reference_link)
        
        # Parse all gazebo elements and complement the corresponding links
        for gazebo_element in root_element.findall('gazebo'):
            reference_name = gazebo_element.get('reference')
            if not reference_name:
                continue

            for sensor_element in gazebo_element.findall('sensor'):
                sensor_type = sensor_element.get('type')
    
                if sensor_type in valid_sensor_types:
                    # CHECK: Is the reference a known link?
                    if reference_name in self.links:
                        target_link = self.links[reference_name]
                        target_link.gazebo_elem = "sensor_" + sensor_type
                        self.sensor_links.append(target_link)
                        print(f"Found sensor {sensor_type} on link {reference_name}")
                    
                    # NEW LOGIC: Is the reference a Joint? 
                    # If so, we find the child link of that joint to attach the sensor metadata
                    else:
                        # Search all links to find which one has this joint_name
                        child_link_of_joint = next((l for l in self.links.values() if l.joint_name == reference_name), None)
                        
                        if child_link_of_joint:
                            child_link_of_joint.gazebo_elem = "sensor_" + sensor_type
                            self.sensor_links.append(child_link_of_joint)
                            print(f"Found sensor {sensor_type} on joint {reference_name}, attributing to child link {child_link_of_joint.name}")
                        else:
                            print(f"Warning: Gazebo reference {reference_name} is neither a known link nor a known joint.")
                else:
                    print("Unsupported sensor type: ", sensor_type, " on reference ", reference_name)
                
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
    
    def get_neighbors(self, link : URDFLink):
        neighbors = []

        parent = link.parent
        if parent is not None:
            neighbors.append(parent)

        children = link.children
        if children is not None:
            neighbors.extend(children)

        return neighbors

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
            'joint_name': link.joint_name
        }
        
    def build_frame_context_hierarchy(self, link: URDFLink) -> Dict:
        """Build comprehensive context about the frame for analysis"""
        return {
            'link': link.name,
            'parent': link.parent.name,
            'joint_type': link.joint_to_parent_type,
            'joint_name': link.joint_name
        }
    
    def build_hierarchy_from_leaf(self, link: URDFLink) -> str:
        example = "head_link --(camera_mount_joint: fixed)--> camera_link"
        # ex:  parent_link --[joint_name:joint_type]--> child_link
        elem = ""
        if(link.parent != None):
            joint = f" --[{link.joint_name}:{link.joint_to_parent_type}]--> "
            elem = link.parent.name + joint + link.name
        return elem
    
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

    def render_tree(
        self,
        start_link: Optional[str] = None,
        max_depth: Optional[int] = None,
        show_joint_info: bool = False
    ) -> str:
        """
        Returns an explicit ASCII tree representation of the URDF.

        Parameters:
            start_link: name of link to start from (None = root)
            max_depth: limit recursion depth (None = full depth)
            show_joint_info: include joint name/type annotations
        """

        if start_link:
            root = self.links.get(start_link)
            if root is None:
                raise ValueError(f"Link '{start_link}' not found in URDF.")
        else:
            if self.root is None:
                raise ValueError("URDF tree has no root.")
            root = self.root

        lines = []

        def _render(node, prefix="", is_last=True, depth=0):
            if max_depth is not None and depth > max_depth:
                return

            connector = "└── " if is_last else "├── "

            if depth == 0:
                lines.append(node.name)
            else:
                lines.append(prefix + connector + node.name)

            if show_joint_info and node.joint_name:
                lines[-1] += f"  [joint: {node.joint_name}, type: {node.joint_to_parent_type}]"
                
            child_count = len(node.children)

            for i, child in enumerate(node.children):
                is_child_last = (i == child_count - 1)

                if depth == 0:
                    new_prefix = ""
                else:
                    new_prefix = prefix + ("    " if is_last else "│   ")

                _render(child, new_prefix, is_child_last, depth + 1)

        _render(node = root)
        return "\n".join(lines)
     
    def render_tree_json(
        self,
        start_link: Optional[str] = None,
        max_depth: Optional[int] = None,
        include_joint_info: bool = False
    ) -> dict:

        if start_link:
            root = self.links.get(start_link)
            if root is None:
                raise ValueError(f"Link '{start_link}' not found.")
        else:
            if self.root is None:
                raise ValueError("URDF tree has no root.")
            root = self.root

        def _build(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return None

            node_dict = {
                "name": node.name,
                "children": []
            }

            if include_joint_info and node.joint_name:
                node_dict["joint"] = {
                    "name": node.joint_name,
                    "type": node.joint_to_parent_type
                }

            for child in node.children:
                child_dict = _build(child, depth + 1)
                if child_dict is not None:
                    node_dict["children"].append(child_dict)

            return node_dict

        return _build(root)  
    
    def get_ancestors(self, link: URDFLink):
        ancestors = []

        current = link
        while current is not None:
            ancestors.append(current)
            current = current.parent

        return ancestors

    def find_lowest_common_ancestor(self, links: List[URDFLink]) -> URDFLink:
        """
        Returns the lowest common ancestor of given links.
        """

        ancestor_sets = [set(self.get_ancestors(link))for link in links]
        common = set.intersection(*ancestor_sets)

        # Pick the deepest one = farthest from root
        def depth(link):
            d = 0
            while link.parent is not None:
                link = link.parent
                d += 1
            return d

        return max(common, key=depth)
    
    def find_highest_common_ancestor(self, links: list[URDFLink]) -> URDFLink:
        """
        Returns the highest common ancestor among a list of URDFLink nodes.
        """

        # Get ancestor chains and reverse to root -> leaf
        ancestor_chains = [list(reversed(self.get_ancestors(link))) for link in links]

        # Walk down from root
        hca = ancestor_chains[0][0]  # start with root
        for i in range(len(ancestor_chains[0])):
            candidate = ancestor_chains[0][i]
            if all(len(chain) > i and chain[i] == candidate for chain in ancestor_chains):
                hca = candidate
            else:
                break

        return hca

    def get_path_from_ancestor(self, ancestor: URDFLink, descendant: URDFLink):
        path = []
        current = descendant

        while current != ancestor:
            path.append(current)
            current = current.parent

        path.append(ancestor)
        return list(reversed(path))
    
    def get_component_nodes(self, leaf_links: List[URDFLink]):

        lca = self.find_lowest_common_ancestor(leaf_links)
        nodes = set()

        for leaf in leaf_links:
            path = self.get_path_from_ancestor(lca, leaf)
            nodes.update(path)

        return lca, nodes
    
    def collect_component_subtree(self, root: URDFLink):

        nodes = set()

        def dfs(link):
            nodes.add(link)

            for child in link.children:
                dfs(child)

        dfs(root)
        return nodes
    
    def build_subtree_dict(self, root: URDFLink, allowed_nodes: set):

        def recurse(node):

            return {
                "name": node.name,
                "joint_type": node.joint_to_parent_type,
                "children": [
                    recurse(child)
                    for child in node.children
                    if child in allowed_nodes
                ]
            }

        return recurse(root)
    
    def extract_component_subtree(self, leaf_links: List[URDFLink]):

        lca, nodes = self.get_component_nodes(leaf_links)
        subtree = self.build_subtree_dict(lca, nodes)

        return subtree, lca
    
    def extract_component_subtree(self, leaf_links_str: List[str]):
        leaf_links = [self.get_link(link) for link in leaf_links_str]
        lca = self.find_lowest_common_ancestor(leaf_links)
        nodes = self.collect_component_subtree(lca)

        return self.build_subtree_dict(lca, nodes), lca
    
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

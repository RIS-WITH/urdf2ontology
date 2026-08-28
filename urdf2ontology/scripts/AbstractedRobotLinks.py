from urdf_tree import URDFLink, URDFTree
from typing import List, Any, Dict, List, Set, Optional
from dataclasses import dataclass, field

@dataclass 
class RobotComponent:
    """Used for all components (upper assembly or sub-components), contains an identifier (camera_0), a type (Camera) and an attachment root being a link (head_link), links contain the ordered subchain representing the component (["camera_link", "camera_link_optical"])"""
    id: str = ""                      # component identifier (e.g., segment_1_3)
    semantic_name: str = ""           # component name after naming task (e.g., left_arm)
    type: str = ""                    # ontology type
    attachment_root: str = ""         # structural boundary between abstracted component and rest of the robot's composition
    link_chain: List[str] = field(default_factory=list) # Subchain representing the abstracted component (ordered sequence)
    textual_kinematic_sequence: List[str] = field(default_factory=list) # Textual version of the subchain ( parent_link --> (joint_name:type) --> child_link )
    dict_kinematic_sequence: List[Dict[str, Any]] = field(default_factory=list) # Dict version of the subchain (with parent, joint and child arguments)
    csv_kinematic_sequence: List[Dict[str, Any]] = field(default_factory=list) # CSV kinematic sequences
    
    def get_chain_dict(self):
        return {self.id: self.link_chain}
    
    def __str__(self) -> str:
        chain_preview = " -> ".join(self.link_chain)
        return (f"[{self.type}] ID: {self.id}\n"
                f"  └─ Root: {self.attachment_root}\n"
                f"  └─ Semantic Name: {self.semantic_name}\n"
                f"  └─ Link Chain: {chain_preview or 'Empty'}")
        
    def get_kinematic_chain(self, mode):
        if mode == "links":
            return self.link_chain
        elif mode == "kinematic_dict":
            return self.dict_kinematic_sequence
        elif mode == "kinematic_txt":
            return self.textual_kinematic_sequence
        elif mode == "kinematic_csv":
            return self.csv_kinematic_sequence
        else:
            print(f"Warning: Unknown operational mode '{mode}' encountered.")
            return 
    
@dataclass  
class SegmentElement(RobotComponent):
    """Used for sub sequential components (limbs), adds the upper and end joint for representing motion planning chains"""
    segment_upper_joint: str = ""    # The joint connecting this to the previous segment
    segment_end_link: str = ""       # The last link in this specific segment
    
    def __str__(self) -> str:
        base_info = super().__str__()
        return (f"{base_info}\n"
                f"  └─ Segment: Joint({self.segment_upper_joint}) -> End Link({self.segment_end_link})")

@dataclass
class ChainComponent(RobotComponent):
    """Contains components which are chain-like (limbs with segments) -> sequential."""
    root_joint: str = ""             # e.g., "shoulder_pan_joint"
    assembly_base_link: str = ""     # e.g., "r_shoulder_link"
    end_link: str = ""               # e.g., "r_gripper_frame_link"
    segments: List[SegmentElement] = field(default_factory=list)  # ordered sequence of functional segments
   
    all_links: set[str] = field(default_factory=set)
    all_joints: set[str] = field(default_factory=set)

    def format_structure(self, indent: str = "        ") -> str:
        """Formats the chain component and its ordered segments for tree visualization."""
        lines = [f"{indent}└─{self.id}: ({self.type})"]
        if self.segments:
            for seg in self.segments:
                seg_type = getattr(seg, "type", "Segment")
                semantic = f" ({seg.semantic_name})" if getattr(seg, "semantic_name", "") else ""
                lines.append(f"{indent}    ├─{seg.id}:{semantic} ({seg_type}) {seg.link_chain}")
        else:
            lines.append(f"{indent}    ├─ Links: {self.link_chain}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        header = f"  CHAIN COMPONENT: {self.id} ({self.type})"
        body = (f"  Attachment Root: {self.attachment_root}\n"
                f"  Joint: {self.root_joint}\n"
                f"  Base Link: {self.assembly_base_link}\n"
                f"  End Link: {self.end_link}\n"
                f"  Segments ({len(self.segments)}):")
        
        seg_info = ""
        for i, seg in enumerate(self.segments):
            indented_seg = str(seg).replace("\n", "\n      ")
            seg_info += f"\n    ({i}) {indented_seg}"
            
        return f"{header}\n{body}{seg_info}"

@dataclass    
class TreeComponent(RobotComponent):
    """Contains components which are tree-like (end-effector & sensor assemblies with sub-parts) -> parallel."""
    mounting_joint: str = ""           # Joint connecting assembly to bodypart
    assembly_base_link: str = ""       # First link of assembly (e.g. "arm_base_link")
    
    branches: List[RobotComponent] = field(default_factory=list)

    all_links: set[str] = field(default_factory=set)
    all_joints: set[str] = field(default_factory=set)
           
    def __str__(self) -> str:
        header = f" TREE COMPONENT: {self.id} ({self.type})"
        body = (f"  Attachment Root: {self.attachment_root}\n"
                f"  Semantic Name: {self.semantic_name}\n"
                f"  Mounting Joint: {self.mounting_joint}\n"
                f"  Base Link: {self.assembly_base_link}\n"
                f"  Structure:")
        
        branch_lines = []
        for branch in self.branches:
            if hasattr(branch, "format_structure"):
                # Renders ChainComponent with its nested segments
                branch_lines.append(branch.format_structure(indent="        "))
            else:
                # Fallback for direct segment objects
                b_semantic = f" ({branch.semantic_name})" if getattr(branch, "semantic_name", "") else ""
                branch_lines.append(f"        └─{branch.id}:{b_semantic} ({branch.type}) {getattr(branch, 'link_chain', [])}")
            
        sub_info = "\n" + "\n".join(branch_lines) if branch_lines else " (No sub-elements)"
        return f"{header}\n{body}{sub_info}"

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
        
        self.urdf_groups = {} # after the pass {'sensor_0': URDFSubTree}
        
        # used for the new version with dataclass structures instead of dicts
        self.components = {} # contains RobotComponent elements
        self.merged_components = {} # contains either TreeComponent or ChainComponent
        
        self.sensor_components = {}
        self.end_effector_components = {}
        self.bodypart_components = {}
        
    def find_lca_from_paths(self, paths: List[List[str]]) -> str:
        """
        Finds the Lowest Common Ancestor from a list of chains.
        """
        if not paths or not paths[0]:
            return None
        
        # 1. Start with the first path as a reference
        reference_path = paths[0]
        lca = None
        
        # 2. Compare each index across all paths
        for i in range(len(reference_path)):
            current_node = reference_path[i]
            
            # Check if this node exists at index i in ALL other paths
            for other_path in paths[1:]:
                # If a path is shorter or has a different node, we've hit the divergence
                if i >= len(other_path) or other_path[i] != current_node:
                    return lca
            
            # All paths share this node
            lca = current_node
                
        return lca

    def create_component_links(self, result_identif, component_type = "branch"):
        cpt = 0
        for identified_component in result_identif:
            if(component_type == "chain"):
                new_id = "chain_" + str(cpt)
                new_component = ChainComponent(id=new_id, end_link=identified_component)
            elif(component_type == "tree"):
                new_id = "tree_" + str(cpt)
                new_component = TreeComponent(id=identified_component)
            elif(component_type == "branch"):
                new_component = RobotComponent(id=identified_component)
                new_id = identified_component
            elif(component_type == "segment"):
                new_component = SegmentElement(id=identified_component)
                new_id = identified_component
                
            new_component.link_chain = self.initial_urdf_tree.get_tree_from_leaf(identified_component)
            # Add the kinematic sequence inputs (link + joint names/types)
            for ln in new_component.link_chain[1:]: # remove first link as it has no parent 
                link = self.initial_urdf_tree.get_link(ln)
                new_component.dict_kinematic_sequence.append(self.initial_urdf_tree.build_link_element(link))
                new_component.textual_kinematic_sequence.append(self.initial_urdf_tree.build_textual_link_element(link))
                new_component.csv_kinematic_sequence.append(self.initial_urdf_tree.build_csv_hierarchy_from_leaf(link))

            self.components.update({new_id: new_component}) # or bodypart_components
            cpt += 1
        
    def add_attachment_root_components(self, result_mounting: Dict[str, str]):
        for link, component in self.components.items():
            mounting_point = result_mounting.get(link, "")
            if(mounting_point != ""):
                component.attachment_root = mounting_point
                idx = component.link_chain.index(mounting_point) # get the index of the attachment root
                component.assembly_base_link = component.link_chain[idx+1] # first link after the mounting point
                self.trim_component_below_root(component)

    def trim_component_below_root(self, component: RobotComponent) -> RobotComponent:
        root = component.attachment_root 

        if root in component.link_chain:
            idx = component.link_chain.index(root)
            component.link_chain = component.link_chain[idx + 1:] 

        csv_idx = next((i for i, row in enumerate(component.csv_kinematic_sequence) if row[0] == root), None)
        if csv_idx is not None:
            component.csv_kinematic_sequence = component.csv_kinematic_sequence[csv_idx:]

        dict_idx = next((i for i, entry in enumerate(component.dict_kinematic_sequence) if entry.get("parent") == root), None)
        if dict_idx is not None:
            component.dict_kinematic_sequence = component.dict_kinematic_sequence[dict_idx:]

        txt_idx = next((i for i, text in enumerate(component.textual_kinematic_sequence) if text.startswith(root)), None)
        if txt_idx is not None:
            component.textual_kinematic_sequence = component.textual_kinematic_sequence[txt_idx:]

        return component

    def add_preclassification_components(self, link_states):
        for comp_id, comp in self.components.items():
            classified_type = link_states.get(comp_id, "")
            if(classified_type != ""):
                comp.type = classified_type
            else:
                print(f"Warning: The component {comp_id} has no classified type.")
                    
        for link, concept in link_states.items():
            self.components[link].type = concept
        
    def create_grouped_components(self, result_grouping, keyword):
        for i, component_links in enumerate(result_grouping):
            component_id = f"{keyword}_{i}"
            new_upper_component = TreeComponent(id=component_id, type="", attachment_root="")
            all_chains = [] 
            attachment_root = ""
            for sublink in component_links:
                selected_subcomponent = self.components[sublink]
                if(attachment_root == ""):
                    attachment_root = selected_subcomponent.attachment_root
                elif(attachment_root != selected_subcomponent.attachment_root):
                    print("Warning: Different attachment roots found between grouped links ->", attachment_root, "!=", selected_subcomponent.attachment_root)
                all_chains.append(selected_subcomponent.link_chain) # right now only supports link_chain but could be extended with txt and dict kinematic
                new_upper_component.branches.append(selected_subcomponent) # add the sub-component as one of the tree branches
    
            new_upper_component.assembly_base_link = self.find_lca_from_paths(all_chains) # get the lca for the upper component
            new_upper_component.attachment_root = attachment_root                                     # update the attachment root as well
            new_upper_component.all_links = {link for chain in all_chains for link in chain}          # output all links in the chains
            self.merged_components.update({component_id: new_upper_component})
        
    def add_postclassification(self, group_states):
        for group, concept in group_states.items():
            self.merged_components[group].type = concept
            
    def create_segments_chain(self, result_segments):
        for chains in result_segments: # loop over subchains
            selected_component = self.components.get(chains["chain_id"])
            chain_number = chains["chain_id"].split("_")[-1]  # get the chain number (e.g. chain_2)
            cpt = 0
            for identified_segment in chains["identified_segments"]:  # loop over segments in each chain
                segment_id = "segment_" + chain_number + "_" + str(cpt) # create a segment id (e.g., segment_2_3)
                segment_links = identified_segment["links"]
                first_link = self.initial_urdf_tree.get_link(segment_links[0]) # get the parent link of the first link in the segement (e.g., torso_link for shoulder_link)

                new_segment = SegmentElement(id=segment_id, type = selected_component.type + "BodyPart", segment_end_link=segment_links[-1])
                if(first_link.parent != None):
                    new_segment.attachment_root = first_link.parent.name
                    new_segment.segment_upper_joint = first_link.joint_name
                    
                new_segment.link_chain = segment_links
                # Update the kinematic sequence inputs
                for ln in new_segment.link_chain[1:]: # remove first link as it has no parent 
                    link = self.initial_urdf_tree.get_link(ln)
                    new_segment.dict_kinematic_sequence.append(self.initial_urdf_tree.build_link_element(link))
                    new_segment.textual_kinematic_sequence.append(self.initial_urdf_tree.build_textual_link_element(link))
                    new_segment.csv_kinematic_sequence.append(self.initial_urdf_tree.build_csv_hierarchy_from_leaf(link))
                
                selected_component.segments.append(new_segment) # add the new segment to the chain component
                for link in segment_links:
                    selected_component.all_links.add(link)
                    # new_component.all_joints. # add the joints?
                cpt += 1
            
    def add_postclassification_segments(self, segment_states):
        for chain_id, chain_comp in self.components.items(): # Loop over the chain components
            for segment_comp in chain_comp.segments:
                classified_type = segment_states.get(segment_comp.id, "")
                if(classified_type != ""):
                    segment_comp.type = classified_type
                else:
                    print(f"Warning: The segment {segment_comp.id} has no classified type.")
    
    def assign_semantic_names(self, result_naming: Dict[str, str]) -> None:
        """
        Assigns semantic names to components, branches, and nested segments.
        
        :param result_naming: Dict mapping component/branch/segment IDs to semantic names.
        :param type: "tree" for merged macro components or "chain" for individual chain components.
        """
        def get_assigned_name(obj_id: str) -> Optional[str]:
            if obj_id in result_naming:
                return result_naming[obj_id]
            # Fallback: check if any key in result_naming ends with the object's ID or matches directly
            for key, val in result_naming.items():
                if key.endswith(f"_{obj_id}") or key == obj_id:
                    return val
            return None

        for comp_id, component in self.merged_components.items():
            if(isinstance(component, TreeComponent)):
                print("Assigning Tree component names:", comp_id)
                tree_name = get_assigned_name(comp_id)
                if tree_name:
                    component.semantic_name = tree_name
                else:
                    print(f"Warning: The tree component '{comp_id}' has no assigned semantic name.")
                
                 # 2. Iterate through branches (ChainComponent or direct branches)
                branches = getattr(component, "branches", [])
                for branch in branches:
                    branch_id = getattr(branch, "id", "")
                    branch_name = get_assigned_name(branch_id)
                    
                    if branch_name:
                        branch.semantic_name = branch_name
                    else:
                        print(f"Warning: Branch '{branch_id}' in tree '{comp_id}' has no assigned semantic name.")

                    # 3. Handle nested segments inside ChainComponent branches
                    segments = getattr(branch, "segments", [])
                    for seg in segments:
                        seg_id = getattr(seg, "id", "")
                        seg_name = get_assigned_name(seg_id)
                        if seg_name:
                            seg.semantic_name = seg_name
                        else:
                            print(f"Warning: Segment '{seg_id}' in branch '{branch_id}' has no assigned semantic name.")
                
            elif(isinstance(component, ChainComponent)):
                print("Assigning Chain component names:", comp_id)
                chain_name = get_assigned_name(comp_id)
                if chain_name:
                    component.semantic_name = chain_name

                    # Process inner segments for individual chains
                    segments = getattr(component, "segments", [])
                    for segment in segments:
                        seg_id = getattr(segment, "id", "")
                        seg_name = get_assigned_name(seg_id)
                        if seg_name:
                            segment.semantic_name = seg_name
                        else:
                            print(f"Warning: Segment '{seg_id}' in chain '{comp_id}' has no assigned semantic name.")
                else:
                    print(f"Warning: The chain component '{comp_id}' has no assigned semantic name.")
                 
    def create_segments_chain_canonical(self, result_segments: list) -> list:
        """
        Creates canonical SegmentElement instances by collapsing identical link chains.
        Aggregates parent chain types, mount links, and links them to their upper components.
        """
        canonical_segments = {}  # Map: tuple(segment_links) -> SegmentElement
        segment_counter = 0

        for chains in result_segments:  # Loop over subchains
            selected_component = self.components.get(chains["chain_id"])

            for identified_segment in chains["identified_segments"]:  # Loop over segments in chain
                segment_links = identified_segment["links"]
                chain_key = tuple(segment_links)

                # Extract upper joint and mount link (attachment root)
                first_link = self.initial_urdf_tree.get_link(segment_links[0])
                attachment_root = first_link.parent.name if (first_link and first_link.parent) else None
                segment_upper_joint = first_link.joint_name if (first_link and first_link.parent) else None

                if chain_key not in canonical_segments:
                    # 1. Create primary canonical SegmentElement
                    segment_id = f"segment_{segment_counter}"
                    segment_counter += 1

                    new_segment = SegmentElement(
                        id=segment_id,
                        type=selected_component.type + "BodyPart",
                        segment_end_link=segment_links[-1]
                    )

                    if attachment_root:
                        new_segment.attachment_root = attachment_root
                        new_segment.segment_upper_joint = segment_upper_joint

                    new_segment.link_chain = segment_links

                    # Track multiple mount links and upper chain types across overlapping chains
                    new_segment.all_attachment_roots = {attachment_root} if attachment_root else set()
                    new_segment.parent_chain_types = {selected_component.type} if hasattr(selected_component, 'type') else set()

                    # Build kinematic sequence details
                    for ln in new_segment.link_chain[1:]:
                        link = self.initial_urdf_tree.get_link(ln)
                        new_segment.dict_kinematic_sequence.append(self.initial_urdf_tree.build_link_element(link))
                        new_segment.textual_kinematic_sequence.append(self.initial_urdf_tree.build_textual_link_element(link))
                        new_segment.csv_kinematic_sequence.append(self.initial_urdf_tree.build_csv_hierarchy_from_leaf(link))

                    canonical_segments[chain_key] = new_segment
                else:
                    # 2. Retrieve existing canonical instance and aggregate incoming metadata
                    new_segment = canonical_segments[chain_key]

                    if attachment_root:
                        new_segment.all_attachment_roots.add(attachment_root)
                    if hasattr(selected_component, 'type') and selected_component.type:
                        new_segment.parent_chain_types.add(selected_component.type)

                # 3. Attach canonical instance to component without duplicates
                if new_segment not in selected_component.segments:
                    selected_component.segments.append(new_segment)

                for link in segment_links:
                    selected_component.all_links.add(link)

        return list(canonical_segments.values())
    
    def apply_generated_names(self, generated_names: Dict[str, str], components) -> None:
        """
        Applies the aggregated semantic names to components in self.merged_components 
        and updates internal self.components references and child sub-structures in place.
        
        :param generated_names: Mapping of { entity_id: new_semantic_name } 
                                from ComponentLinkNamingTask.aggregate()
        """
        # 1. Update merged_components (Trees and standalone Chains)
        for macro_id, macro_comp in list(components.items()):
            if macro_id in generated_names:
                new_macro_name = generated_names[macro_id]
                
                # Apply to name attribute (or id/type depending on model requirements)
                if hasattr(macro_comp, "name"):
                    macro_comp.name = new_macro_name
                macro_comp.type = new_macro_name

            # Handle TreeComponent branches
            if isinstance(macro_comp, TreeComponent) or hasattr(macro_comp, "branches"):
                for branch in getattr(macro_comp, "branches", []):
                    branch_id = getattr(branch, "id", "")
                    if branch_id in generated_names:
                        new_branch_name = generated_names[branch_id]
                        if hasattr(branch, "name"):
                            branch.name = new_branch_name
                        branch.type = new_branch_name

                    # Handle segments inside TreeComponent branches
                    for seg in getattr(branch, "segments", []):
                        seg_id = getattr(seg, "id", "")
                        if seg_id in generated_names:
                            new_seg_name = generated_names[seg_id]
                            if hasattr(seg, "name"):
                                seg.name = new_seg_name
                            seg.type = new_seg_name

            # Handle standalone ChainComponent segments
            elif isinstance(macro_comp, ChainComponent) or hasattr(macro_comp, "segments"):
                for seg in getattr(macro_comp, "segments", []):
                    seg_id = getattr(seg, "id", "")
                    if seg_id in generated_names:
                        new_seg_name = generated_names[seg_id]
                        if hasattr(seg, "name"):
                            seg.name = new_seg_name
                        seg.type = new_seg_name

        # # 2. Synchronize self.components (Direct references)
        # for chain_id, chain_comp in self.components.items():
        #     if chain_id in generated_names:
        #         new_name = generated_names[chain_id]
        #         if hasattr(chain_comp, "name"):
        #             chain_comp.name = new_name
        #         chain_comp.type = new_name

        #     # Update segments inside self.components
        #     for seg in getattr(chain_comp, "segments", []):
        #         seg_id = getattr(seg, "id", "")
        #         if seg_id in generated_names:
        #             new_seg_name = generated_names[seg_id]
        #             if hasattr(seg, "name"):
        #                 seg.name = new_seg_name
        #             seg.type = new_seg_name
        
class SensorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)

class ActuatorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)

class EndEffectorLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
        
class BodyPartLinks(ComponentLinks):
    def __init__(self, urdf_tree: URDFTree):
        super().__init__(urdf_tree)
        

class AbstractedRobotLinks:
    def __init__(self, name, path, urdf_tree):
        self.robot_name = name
        self.urdf_path = path
        self.urdf_tree = urdf_tree
        self.sensor_links = SensorLinks(urdf_tree)
        self.actuator_links = ActuatorLinks(urdf_tree)
        self.end_effector_links = EndEffectorLinks(urdf_tree)
        self.body_part_links = BodyPartLinks(urdf_tree)
        
        self.sensor_groups = Dict[str, URDFTree]
        self.end_effector_groups = Dict[str, URDFTree]
        self.actuated_locomotion_groups = Dict[str, URDFTree]
        self.actuated_bodypart_groups = Dict[str, URDFTree]
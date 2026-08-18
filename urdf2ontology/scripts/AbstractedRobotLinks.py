from urdf_tree import URDFLink, URDFTree
from typing import List, Any, Dict, List, Set
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
class ChainComponent(RobotComponent): # Contains 
    """Contains components which are chain-like (limbs with segments) -> sequential."""
    root_joint: str = ""             # e.g., "shoulder_pan_joint" (joint between the attachment root and the first link)
    assembly_base_link: str = ""     # e.g., "r_shoulder_link" (first link in the segments)
    end_link: str = ""               # e.g., "r_gripper_frame_link" (last link the sequence)
    segments: List[SegmentElement] = field(default_factory=list)  # ordered sequence of functional segments
   
    all_links: set[str] = field(default_factory=set)
    all_joints: set[str] = field(default_factory=set)
    
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
    """Contains components which are tree-like (end-effector & sensor assemblies with sub-parts) -> paralell."""
    mounting_joint: str = ""           # Joint connecting the assembly to the bodypart (e.g., "wrist_roll_joint" or "head_camera_joint")
    assembly_base_link: str = ""       # The first link of the component assembly (e.g., "r_gripper_link" or "head_camera_link")
    
    branches: List[RobotComponent] = field(default_factory=list) # contains all sub-components that are part of an assembly unit

    # Global list of all links and joints involved in the tree
    all_links: set[str] = field(default_factory=set)
    all_joints: set[str] = field(default_factory=set)
           
    def __str__(self) -> str:
        header = f" TREE COMPONENT: {self.id} ({self.type})"
        body = (f"  Attachment Root: {self.attachment_root}\n"
                f"  Semantic Name: {self.semantic_name}\n"
                f"  Mounting Joint: {self.mounting_joint}\n"
                f"  Base Link: {self.assembly_base_link}\n"
                f"  Structure:")
        
        sub_info = ""
        for branch in self.branches:
            sub_info += f"\n\t└─{branch.id}: {branch.semantic_name} ({branch.type}) {branch.link_chain}"
            
        return f"{header}\n{body}{sub_info or ' (No sub-elements)'}"

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
        
    def find_group_lca(self, group_links, trimmed_chains):
        """
        Finds the Lowest Common Ancestor for a group of links.
        """
        if not group_links:
            return None
        
        # 1. Get the full paths for all links in the group
        # We need the full path from Robot Root to the specific Link
        paths = []
        for link in group_links:
            if link in trimmed_chains:
                # We reconstruct the path: [root, ..., parent, link]
                path = trimmed_chains[link]['kinematic_chain'] + [link]
                paths.append(path)
        
        if not paths:
            return group_links[0] # Fallback

        # 2. Find the shortest path to avoid IndexErrors
        min_length = min(len(p) for p in paths)
        
        lca = paths[0][0] # Start with the absolute root (e.g., base_link)
        
        # 3. Iterate through the paths index by index
        for i in range(min_length):
            # Get the link at index 'i' for the first path
            current_link = paths[0][i]
            
            # Check if this link is the same at this index for ALL paths
            if all(p[i] == current_link for p in paths):
                lca = current_link
            else:
                # The paths have diverged! The previous 'lca' was the last common one.
                break
                
        return lca  
    
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
                # Trim each of the subchain from after the attachment root to their leaf link
                component.link_chain = component.link_chain[idx+1:]
                component.dict_kinematic_sequence = component.dict_kinematic_sequence[idx+1:]
                component.textual_kinematic_sequence = component.textual_kinematic_sequence[idx+1:]
                component.csv_kinematic_sequence = component.csv_kinematic_sequence[idx+1:]
            # else: # if the output did not generate a mounting point for that link
            #     print(f"Warning! -> The Link {link} has not been processed as having an attachment root.") 
                # component.attachment_root = component.link_chain[-2] # get the link just above the leaf link
                # component.assembly_base_link = link
  
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
    
    def assign_semantic_names(self, result_naming, type = "tree"):
        if(type == "tree"):
            for tree_id, tree_comp in self.merged_components.items():
                if(tree_id in result_naming.keys()):
                    tree_comp.semantic_name = result_naming[tree_id]
                    for branch in tree_comp.branches:
                        if(branch.id in result_naming.keys()):
                            branch.semantic_name = result_naming[branch.id]
                else:
                    print(f"Warning: The tree {tree_id} has no assigned name.")
        elif(type == "chain"):
            for chain_id, chain_comp in self.components.items(): 
                if(chain_id in result_naming.keys()):
                    chain_comp.semantic_name = result_naming[chain_id]
                    for segment in chain_comp.segments:
                        if(segment.id in result_naming.keys()):
                            segment.semantic_name = result_naming[segment.id]
                else:
                    print(f"Warning: The chain {chain_id} has no assigned name.")
                    
    def assert_names_into_components(
        merged_components: Dict[str, Any], raw_llm_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Flattens chain-based LLM output and assigns semantic names to macro entities

        and their internal SegmentElement objects.
        """
        generated_names = raw_llm_output.get("generated_names", {})

        # 1. Build a unique lookup table for segment names (deduplicating across chains)
        segment_name_map: Dict[str, str] = {}
        chain_name_map: Dict[str, str] = {}

        for chain_id, chain_data in generated_names.items():
            comp_name = chain_data.get("component_name")
            sub_comps = chain_data.get("sub_components", {})

            for seg_id, seg_name in sub_comps.items():
                if seg_id not in segment_name_map:
                    segment_name_map[seg_id] = seg_name

        # 2. Iterate through macro entities and update segments
        for entity_id, entity_data in merged_components.items():
            segments = (
                entity_data.get("segments", [])
                if isinstance(entity_data, dict)
                else getattr(entity_data, "segments", [])
            )

            matched_seg_names = []

            for seg in segments:
                # Extract segment ID regardless of whether it's a dict or dataclass/object
                seg_id = (
                    seg.get("id") if isinstance(seg, dict) else getattr(seg, "id", None)
                )

                if seg_id and seg_id in segment_name_map:
                    assigned_name = segment_name_map[seg_id]

                    # Assert name onto segment object or dictionary
                    if isinstance(seg, dict):
                        seg["semantic_name"] = assigned_name
                    else:
                        seg.semantic_name = assigned_name

                    matched_seg_names.append(assigned_name)

            # 3. Assign macro entity semantic name based on macro type and assigned segments
            macro_type = (
                entity_data.get("macro_type", "Component")
                if isinstance(entity_data, dict)
                else getattr(entity_data, "type", "Component")
            )

            # Build clean entity name from primary segment or macro type
            if matched_seg_names:
                primary_name = matched_seg_names[0]
                entity_semantic_name = (
                    f"{primary_name}_assembly"
                    if "assembly" not in primary_name
                    else primary_name
                )
            else:
                entity_semantic_name = macro_type.lower()

            if isinstance(entity_data, dict):
                entity_data["semantic_name"] = entity_semantic_name
            else:
                entity_data.semantic_name = entity_semantic_name

        print(
            f"Successfully asserted names for {len(segment_name_map)} unique segments across {len(merged_components)} macro entities."
        )
        return merged_components
                    
    # def build_macro_entities_dict(self, grouped_entities: list) -> dict:
    #     """
    #     Maps the grouped LLM output (List[List[List[str]]]) back to SegmentElement instances.
    #     Deduplicates segments into a single macro entity dictionary.
    #     """
    #     # 1. Build a reverse lookup from tuple(link_chain) -> (SegmentElement, ParentComponent)
    #     segment_lookup = {}
    #     for comp_id, component in self.components.items():
    #         for segment in component.segments:
    #             # Map exact link chain tuple to the SegmentElement
    #             segment_lookup[tuple(segment.link_chain)] = (segment, component)

    #     macro_entities_dict = {}

    #     # 2. Process each grouped entity (macro body part)
    #     for idx, entity_subchains in enumerate(grouped_entities):
    #         seen_segments = {}          # Map of segment_id -> SegmentElement (deduplicated)
    #         all_entity_links = set()    # Set of all URDF links in this macro entity
    #         attachment_roots = set()    # Set of mount links / attachment roots
    #         lower_segment_types = set() # Set of lower-level segment types
    #         parent_chain_types = set()  # Set of parent component types

    #         for subchain_links in entity_subchains:
    #             chain_key = tuple(subchain_links)
                
    #             # Lookup segment instance by exact link sequence
    #             if chain_key in segment_lookup:
    #                 segment, parent_comp = segment_lookup[chain_key]
                    
    #                 # Deduplicate segment by ID
    #                 if segment.id not in seen_segments:
    #                     seen_segments[segment.id] = segment
    #                     lower_segment_types.add(segment.type)
                        
    #                     if hasattr(segment, 'attachment_root') and segment.attachment_root:
    #                         attachment_roots.add(segment.attachment_root)
                        
    #                     if hasattr(parent_comp, 'type') and parent_comp.type:
    #                         parent_chain_types.add(parent_comp.type)

    #                 all_entity_links.update(segment.link_chain)
    #             else:
    #                 # Fallback: Match individual links if segment boundary changed slightly
    #                 all_entity_links.update(subchain_links)

    #         # 3. Derive primary entity metadata
    #         # Resolve primary attachment root (common root or first encountered)
    #         primary_mount = next(iter(attachment_roots)) if attachment_roots else "unknown_root"
            
    #         # Resolve primary macro component type
    #         primary_type = next(iter(parent_chain_types)) if parent_chain_types else "MacroBodyPart"
            
    #         entity_id = f"{primary_type.lower()}_entity_{idx}"

    #         # 4. Store consolidated entity in the new container
    #         macro_entities_dict[entity_id] = {
    #             "entity_id": entity_id,
    #             "macro_type": primary_type,
    #             "mount_link": primary_mount,
    #             "all_mount_links": list(attachment_roots),
    #             "lower_level_segment_types": list(lower_segment_types),
    #             "all_links": sorted(list(all_entity_links)),
    #             "segments": list(seen_segments.values()) # Deduplicated list of SegmentElement instances
    #         }

    #     return macro_entities_dict
    
    # def build_macro_entities_dict(self, grouped_entities: list) -> dict:
    #     macro_entities_dict = {}

    #     for idx, entity_subchains in enumerate(grouped_entities):
    #         seen_segments = {}
    #         all_entity_links = set()
    #         attachment_roots = set()
    #         lower_segment_types = set()
    #         parent_chain_types = set()

    #         for subchain_links in entity_subchains:
    #             subchain_set = set(subchain_links)
    #             all_entity_links.update(subchain_links)
                
    #             # Find any SegmentElement that shares links with this subchain
    #             matched = False
    #             for comp_id, component in self.components.items():
    #                 for segment in component.segments:
    #                     # Match if the segment's links overlap with the subchain links
    #                     if set(segment.link_chain).intersection(subchain_set):
    #                         matched = True
    #                         if segment.id not in seen_segments:
    #                             seen_segments[segment.id] = segment
    #                             lower_segment_types.add(segment.type)
                                
    #                             if getattr(segment, 'attachment_root', None):
    #                                 attachment_roots.add(segment.attachment_root)
    #                             if getattr(component, 'type', None):
    #                                 parent_chain_types.add(component.type)

    #         primary_mount = next(iter(attachment_roots)) if attachment_roots else "unknown_root"
    #         primary_type = next(iter(parent_chain_types)) if parent_chain_types else "MacroBodyPart"
    #         entity_id = f"{primary_type.lower()}_entity_{idx}"

    #         macro_entities_dict[entity_id] = {
    #             "entity_id": entity_id,
    #             "macro_type": primary_type,
    #             "mount_link": primary_mount,
    #             "all_mount_links": list(attachment_roots),
    #             "lower_level_segment_types": list(lower_segment_types),
    #             "all_links": sorted(list(all_entity_links)),
    #             "segments": list(seen_segments.values())
    #         }

    #     return macro_entities_dict
    
    def build_macro_entities_dict(self, grouped_entities: list) -> dict:
        macro_entities_dict = {}

        for idx, entity_subchains in enumerate(grouped_entities):
            seen_segments = {} # Now maps tuple(link_chain) -> SegmentElement
            all_entity_links = set()
            attachment_roots = set()
            lower_segment_types = set()
            parent_chain_types = set()

            for subchain_links in entity_subchains:
                subchain_set = set(subchain_links)
                all_entity_links.update(subchain_links)
                
                for comp_id, component in self.components.items():
                    for segment in component.segments:
                        # Match if the segment's links overlap with the subchain
                        if set(segment.link_chain).intersection(subchain_set):
                            
                            # FIX: Deduplicate by the actual physical links, not the arbitrary ID
                            physical_chain_key = tuple(segment.link_chain)
                            
                            if physical_chain_key not in seen_segments:
                                seen_segments[physical_chain_key] = segment
                                lower_segment_types.add(segment.type)
                                
                                if getattr(segment, 'attachment_root', None):
                                    attachment_roots.add(segment.attachment_root)
                                if getattr(component, 'type', None):
                                    parent_chain_types.add(component.type)

            primary_mount = next(iter(attachment_roots)) if attachment_roots else "unknown_root"
            primary_type = next(iter(parent_chain_types)) if parent_chain_types else "MacroBodyPart"
            entity_id = f"{primary_type.lower()}_entity_{idx}"

            macro_entities_dict[entity_id] = {
                "entity_id": entity_id,
                "macro_type": primary_type,
                "mount_link": primary_mount,
                "all_mount_links": list(attachment_roots),
                "lower_level_segment_types": list(lower_segment_types),
                "all_links": sorted(list(all_entity_links)),
                "segments": list(seen_segments.values()) # Returns the deduplicated segments
            }

        return macro_entities_dict
    
    def canonicalize_segments(all_raw_segments: list) -> dict:
        """
        Collapses identical link chains into a single canonical SegmentElement
        while preserving all mount links and parent chain context.
        """
        canonical_dict = {}  # tuple(link_chain) -> SegmentElement

        for seg in all_raw_segments:
            chain_key = tuple(seg.link_chain)

            if chain_key not in canonical_dict:
                # Store the primary canonical instance
                canonical_dict[chain_key] = seg
                seg.all_attachment_roots = {seg.attachment_root} if hasattr(seg, 'attachment_root') else set()
            else:
                # Merge context (e.g., attachment roots) into the existing canonical segment
                existing_seg = canonical_dict[chain_key]
                if hasattr(seg, 'attachment_root') and seg.attachment_root:
                    existing_seg.all_attachment_roots.add(seg.attachment_root)

        return canonical_dict
    
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
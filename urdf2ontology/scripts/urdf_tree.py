import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional, Set, Any

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
    def __init__(self, name:str ,  urdf_path: str):
        self.robot_name = name
        self.urdf_path = urdf_path
        self.links: Dict[str, URDFLink] = {}
        self.root: Optional[URDFLink] = None
        self.leaf_links: List[URDFLink] = []
        self.sensor_links: List[URDFLink] = [] # used for gazebo plugins
        self.actuator_links: List[URDFLink] = []
        self.end_effector_links: List[URDFLink] = [] # used for gazebo plugins
        self.body_part_links: List[URDFLink] = []
        self._parse_urdf()

    def abstract_components_tree_safe(self, final_trees):
        processed_count = 0
        
        all_classified_nodes = set()
        global_purge_registry = set()
        global_bridge_registry = {} # Maps link_name -> List of group IDs it belongs to
        
        all_attachment_roots = {tree.attachment_root for tree in final_trees.values()}

        # Gather every single node classified across all tree groups
        for tree_comp in final_trees.values():
            all_classified_nodes.update(tree_comp.all_links)

        for gid, tree_comp in final_trees.items():
            mount_name = tree_comp.attachment_root
            full_group_membership = set(tree_comp.all_links)

            if mount_name not in self.links:
                print(f"[SKIP] Designated mount link '{mount_name}' missing from registry for {gid}")
                continue
                
            processed_count += 1

            component_entry_links = []
            for link_name in full_group_membership:
                if link_name in self.links:
                    current_link = self.links[link_name]
                    if current_link.parent and (current_link.parent.name == mount_name or current_link.parent.name not in full_group_membership):
                        component_entry_links.append(current_link)

            for entry_link in component_entry_links:
                downstream_leaves = self._find_downstream_leaves(entry_link)
                
                # Check if this branch leads to any completely unclassified external leaf links
                external_leaves = [leaf for leaf in downstream_leaves if leaf.name not in all_classified_nodes]

                if external_leaves:
                    # WE CANNOT DELETE THIS BRANCH as it contains pathways to unclassified leaf nodes.
                    print(f"[BRIDGE PRESERVATION] Branch '{entry_link.name}' reaches unclassified leaves: {[l.name for l in external_leaves]}. Keeping chain intact.")
                    self._mark_branch_as_bridge(entry_link, all_classified_nodes, gid, global_bridge_registry)
                else:
                    # SAFE TO PURGE: This branch only leads to classified nodes so mark for complete deletion.
                    self._mark_branch_for_purge(entry_link, full_group_membership, global_purge_registry)

        # Step A: Apply metadata annotations to the bridge links remaining in the tree
        for link_name, group_ids in global_bridge_registry.items():
            if link_name in self.links:
                target_link = self.links[link_name]
                
                # Set dynamic attributes directly onto the link object for your ontology parser
                target_link.is_abstracted_bridge = True
                if not hasattr(target_link, "associated_component_groups"):
                    target_link.associated_component_groups = []
                target_link.associated_component_groups.extend(group_ids)
                
                print(f"[TAGGED BRIDGE] Link '{link_name}' kept in tree. Indicated abstraction for groups: {group_ids}")

        # Step B: Execute physical tree deletions (ensuring bridge nodes are protected)
        purged_count = 0
        for dead_node_name in global_purge_registry:
            # Overriding Guard: An unclassified leaf path node can NEVER be deleted, 
            # even if a different component marked it for purging.
            if dead_node_name in global_bridge_registry or dead_node_name in all_attachment_roots:
                continue
                
            if dead_node_name in self.links:
                dead_node = self.links[dead_node_name]
                
                # Sever from active URDF tree
                if dead_node.parent and dead_node in dead_node.parent.children:
                    dead_node.parent.children.remove(dead_node)
                
                dead_node.children.clear()
                del self.links[dead_node_name]
                purged_count += 1

        if processed_count > 0:
            self.refresh_leaf_links()

        print(f"\n[SUCCESS] Tree abstraction complete. Purged {purged_count} group-exclusive links. Tagged {len(global_bridge_registry)} bridge tracking nodes.")
        return {"purged_count": purged_count, "bridges_tagged": list(global_bridge_registry.keys())}

    # =====================================================================
    # HELPER UTILITIES
    # =====================================================================
    def _find_downstream_leaves(self, current_link) -> list:
        """Helper to find all terminal leaf links reachable from this point down the tree."""
        if not current_link.children:
            return [current_link]
        leaves = []
        for child in current_link.children:
            leaves.extend(self._find_downstream_leaves(child))
        return leaves

    def _mark_branch_for_purge(self, current_link, group_membership, purge_set):
        """Recursively flags a clean branch for eventual purging."""
        if current_link.name in group_membership:
            purge_set.add(current_link.name)
        for child in current_link.children:
            self._mark_branch_for_purge(child, group_membership, purge_set)

    def _mark_branch_as_bridge(self, current_link, all_classified_nodes, group_id, bridge_registry):
        """Recursively marks and tags links along paths to unclassified nodes."""
        if current_link.name in all_classified_nodes:
            if current_link.name not in bridge_registry:
                bridge_registry[current_link.name] = []
            if group_id not in bridge_registry[current_link.name]:
                bridge_registry[current_link.name].append(group_id)
                
        for child in current_link.children:
            self._mark_branch_as_bridge(child, all_classified_nodes, group_id, bridge_registry)
    
    def purge_abstracted_components(
        self, 
        final_trees: Dict[str, Any], 
        orphan_links: Optional[Set[str]] = None
    ) -> int:
        """
        Removes all component links and orphan sub-trees from self.links (the URDF tree).
        
        Args:
            final_trees: Dict containing component definitions.
            orphan_links: Optional set of pre-discovered orphan link names to purge.
                          If None, orphans will be automatically discovered during traversal.
                          
        Returns:
            int: Number of links purged from the URDF tree.
        """
        all_purged_nodes: Set[str] = set()

        for gid, tree_comp in final_trees.items():
            component_links = set(getattr(tree_comp, "all_links", []))
            all_purged_nodes.update(component_links)

            # Discover orphans if not explicitly provided
            if orphan_links is None:
                for link_name in component_links:
                    if link_name in self.links:
                        comp_link_obj = self.links[link_name]
                        external_children = [
                            child for child in comp_link_obj.children
                            if child.name not in component_links
                        ]
                        for orphan_link in external_children:
                            self._collect_orphan_subtree_names(
                                orphan_link, component_links, all_purged_nodes
                            )

        if orphan_links:
            all_purged_nodes.update(orphan_links)

        # ---------------------------------------------------------------------
        # PHYSICAL URDF PURGE
        # ---------------------------------------------------------------------
        purged_count = 0
        for dead_node_name in all_purged_nodes:
            if dead_node_name in self.links:
                dead_node = self.links[dead_node_name]

                # Sever parent connection
                if dead_node.parent and dead_node in dead_node.parent.children:
                    dead_node.parent.children.remove(dead_node)

                dead_node.children.clear()
                del self.links[dead_node_name]
                purged_count += 1

        if purged_count > 0:
            self.refresh_leaf_links()

        print(f"[URDF PURGE SUCCESS] Completely removed {purged_count} links from URDF model.")
        return purged_count

    def _collect_orphan_subtree_names(
        self, current_link: Any, component_links: Set[str], purge_set: Set[str]
    ):
        """Recursively collects downstream orphan link names for purging."""
        orphan_name = current_link.name
        if orphan_name in component_links:
            return

        purge_set.add(orphan_name)
        for child_link in list(current_link.children):
            self._collect_orphan_subtree_names(child_link, component_links, purge_set)
            

    def _selective_purge_2(self, link: URDFLink, group_membership: set, destination_target: URDFLink):
        """
        Recursively processes sub-trees. Destroys component links from global registries
        and re-parents unclassified 'stranger' links safely to the destination target.
        """
        # Create a shallow copy snapshot to avoid mutation race-conditions during recursion
        current_children = list(link.children)
        
        for child in current_children:
            if child.name in group_membership:
                # Continue descending into known component territory
                self._selective_purge_2(child, group_membership, destination_target)
            else:
                # Found an unclassified stranger link branching out from within the component sub-tree
                print(f"[WARNING] Unclassified child '{child.name}' detected under component link '{link.name}'.")
                print(f"          Preserving structural continuity: Moving tracking to anchor: '{destination_target.name}'")
                
                # # Sever connection from the component link that is about to be deleted
                # link.children.remove(child)
                
                # # Re-parent the stranger link to keep it connected to the active robot skeleton
                # child.parent = destination_target
                # destination_target.add_child(child)

        # 5. Global Cleanup: Erase the component link out of the master graph registry
        if link.name in group_membership:
            if link.name in self.links:
                del self.links[link.name]
                
    def abstract_components(self, final_groups: Dict, keep_abstracted_links=False):
        for gid, group_data in final_groups.items():
            comp_type = group_data['component_type']
            leaf_names = set(group_data['links'])
            
            first_leaf = next(iter(leaf_names))
            mount_name = group_data['details'][first_leaf]['mount']
            
            if mount_name not in self.links:
                continue
                
            mount_link = self.links[mount_name]

            # Expand membership to include all intermediate links up to the mount
            full_group_membership = set(leaf_names)
            roots_to_prune = set()
            
            for name in leaf_names:
                if name not in self.links: continue
                curr = self.links[name]
                while curr.parent:
                    if curr.parent.name == mount_name:
                        roots_to_prune.add(curr)
                        full_group_membership.add(curr.name)
                        break
                    curr = curr.parent
                    full_group_membership.add(curr.name)

            new_link_name = f"{comp_type}_{gid.split('_')[-1]}"
            new_component = URDFLink(new_link_name) if keep_abstracted_links else None

            # Pass both the potential new_component AND the mandatory mount_link
            for root_link in roots_to_prune:
                self._selective_purge(root_link, full_group_membership, new_component, mount_link)

            # Standard cleanup of the mount's direct children
            root_names = [r.name for r in roots_to_prune]
            mount_link.children = [c for c in mount_link.children if c.name not in root_names]

            if keep_abstracted_links:
                example_root = list(roots_to_prune)[0]
                new_component.joint_to_parent_type = example_root.joint_to_parent_type
                new_component.joint_name = f"joint_{new_link_name}_to_{mount_name}"
            
                mount_link.add_child(new_component)
                self.links[new_link_name] = new_component
                self.leaf_links.append(new_component)
                
            self.refresh_leaf_links()
            
    def _selective_purge(self, link: URDFLink, group_membership: set, new_parent: URDFLink, mount_fallback: URDFLink):
        """
        Purges group members and re-parents 'strangers' to either the new abstraction
        or the original mounting point.
        """
        for child in list(link.children):
            if child.name in group_membership:
                # Recursively purge children that belong to the group
                self._selective_purge(child, group_membership, new_parent, mount_fallback)
            else:
                # Found a 'stranger' (unclassified link)
                print(f"[WARNING] Unclassified child '{child.name}' detected under abstracted link '{link.name}'.")
                print(f"          Moving '{child.name}' to mount: '{mount_fallback.name}'")
                
                link.children.remove(child)
                
                # Destination: Use the new semantic link if it exists, otherwise the original mount
                destination = new_parent if new_parent is not None else mount_fallback
                
                child.parent = destination
                destination.add_child(child)

        # Final deletion of the link from the global registry
        if link.name in group_membership and link.name in self.links:
            print(f"Purging component link: {link.name}")
            del self.links[link.name]
        
    def refresh_leaf_links(self):
        """
        Re-scans the tree to identify links that have no children.
        """
        # A leaf is any link that currently exists and has no children
        self.leaf_links = [
            link for name, link in self.links.items() 
            if len(link.children) == 0
        ]
    
    def abstract_components_tree(self, final_trees, keep_abstracted_links=False):
        """
        Collapses TreeComponents (Sensors/EndEffectors) into single abstracted links.
        
        Args:
            final_groups: Dict mapping group_id to TreeComponent objects.
            keep_abstracted_links: If True, replaces the group with a single 'proxy' link.
        """
        
        components_to_verify = {}
        for gid, tree_comp in final_trees.items():
            # 1. Extract structural metadata from the pre-processed TreeComponent
            comp_type = tree_comp.type
            mount_name = tree_comp.attachment_root
            base_link_name = tree_comp.assembly_base_link
            
            # 2. Ensure the base_link itself is in the membership set so it gets purged
            full_group_membership = set(tree_comp.all_links)
            if base_link_name not in full_group_membership:
                full_group_membership.add(base_link_name)

            # 3. Validation: Check if the structural boundary exists in the current URDF
            if mount_name not in self.links or base_link_name not in self.links:
                print(f"[SKIP] Could not find mount '{mount_name}' or base '{base_link_name}' for {gid}")
                components_to_verify.update({gid: tree_comp})
                continue
                
            mount_link = self.links[mount_name]
            base_link_obj = self.links[base_link_name]

            # 4. Create the Abstracted Proxy Link (The "Semantic" link)
            # Format: e.g., "rgb_camera_0" or "dual_finge_gripper_1"
            new_link_name = f"{comp_type}_{gid.split('_')[-1]}"
            new_component = URDFLink(new_link_name) if keep_abstracted_links else None

            # 5. Perform the Selective Purge 
            # We start at the LCA (base_link_obj). 
            # Strangers (links not in membership) are moved to the new_component or mount.
            self._selective_purge(link=base_link_obj, group_membership=full_group_membership, new_parent=new_component, mount_fallback=mount_link)

            # 6. Detach the original sub-tree root from the mount
            # This breaks the physical connection between the robot and the old assembly base
            mount_link.children = [c for c in mount_link.children if c.name != base_link_name]

            # 7. Finalize the Abstraction (Re-attach the Proxy)
            if keep_abstracted_links:
                # Inherit the kinematic properties from the original connection
                # (e.g., if the camera was on a 'fixed' joint, the proxy stays 'fixed')
                new_component.joint_to_parent_type = base_link_obj.joint_to_parent_type
                new_component.joint_name = f"joint_{new_link_name}_to_{mount_name}"
                
                # Map the new parent-child relationship
                new_component.parent = mount_link
                mount_link.add_child(new_component)
                
                # Register the new proxy link in the global URDF index
                self.links[new_link_name] = new_component
                
            # 8. Update the global leaf link list after structural changes
            self.refresh_leaf_links()

        print(f"Abstraction complete. Processed {len(final_trees)} TreeComponents.")
    
    def abstract_components_chain(self, final_chains, keep_abstracted_links=False):
        """
        Collapses ChainComponents (Limbs, Arms, Legs) into single abstracted macro-links.
        
        Args:
            final_chains: Dict mapping chain_id to ChainComponent objects.
            keep_abstracted_links: If True, replaces the long serial chain with a single 'proxy' link.
        """
        for cid, chain_comp in final_chains.items():
            # 1. Extract structural metadata from the pre-processed ChainComponent
            comp_type = chain_comp.type
            mount_name = chain_comp.attachment_root
            leaf_link_name = chain_comp.end_link
            
            # 2. Gather full membership of the serial path to purge
            full_chain_membership = set(chain_comp.all_links)

            # 3. Validation: Check if the structural boundaries exist in the current URDF
            if mount_name not in self.links or leaf_link_name not in self.links:
                print(f"[SKIP] Could not find mount '{mount_name}' or leaf '{leaf_link_name}' for {cid}")
                continue
                
            mount_link = self.links[mount_name]
            leaf_link_obj = self.links[leaf_link_name]

            # 4. Create the Abstracted Proxy Link (The "Limb" link)
            # Format: e.g., "manipulator_0" or "locomotion_leg_1"
            new_link_name = f"{comp_type}_{cid.split('_')[-1]}"
            new_component = URDFLink(new_link_name) if keep_abstracted_links else None

            # 5. Perform the Selective Purge 
            # We start at the top of the chain (the immediate child of the mount link)
            # and purge down to the leaf link while preserving independent side branches.
            immediate_chain_root = None
            for child in mount_link.children:
                if child.name in full_chain_membership:
                    immediate_chain_root = child
                    break
            
            if not immediate_chain_root:
                print(f"[SKIP] Immediate root link of chain '{cid}' not found under mount '{mount_name}'")
                continue

            self._selective_purge(
                link=immediate_chain_root, 
                group_membership=full_chain_membership, 
                new_parent=new_component, 
                mount_fallback=mount_link
            )

            # 6. Detach the original continuous chain from the mount
            # This breaks the physical connection between the robot and the old serial chain base
            mount_link.children = [c for c in mount_link.children if c.name != immediate_chain_root.name]

            # 7. Finalize the Abstraction (Re-attach the Proxy Link)
            if keep_abstracted_links:
                # Inherit the kinematic properties from the original connection to the mount
                new_component.joint_to_parent_type = immediate_chain_root.joint_to_parent_type
                new_component.joint_name = f"joint_{new_link_name}_to_{mount_name}"
                
                # Map the new parent-child relationship
                new_component.parent = mount_link
                mount_link.add_child(new_component)
                
                # Register the new proxy link in the global URDF index
                self.links[new_link_name] = new_component
                
            # 8. Update the global leaf link list after structural changes
            self.refresh_leaf_links()

        print(f"Abstraction complete. Processed {len(final_chains)} ChainComponents")
        
    def abstract_components(self, final_components, component_type, keep_abstracted_links = False):
        
        if(component_type == "tree"):
            print("Abstracting Tree Components")
            
            # 1. Isolate the Component Entry Nodes: Identify the highest links within the component's membership whose parents live outside the component.
            for tree_id, tree_comp in final_components.items():
                print(f"Abstracting component {tree_comp}")
                mount_name = tree_comp.attachment_root
                base_link_name = tree_comp.assembly_base_link
                
                # full_group_membership = set(tree_comp.all_links)
                # if base_link_name not in full_group_membership: # check that the base_link is in the group
                #     full_group_membership.add(base_link_name)
                    
                mount_link = self.links[mount_name]
                base_link_obj = self.links[base_link_name]
                
                 # 4. Create the Abstracted Proxy Link (The "Semantic" link)
                # Format: e.g., "rgb_camera_0" or "dual_finge_gripper_1"
                new_component = URDFLink(tree_comp.semantic_name) if keep_abstracted_links else None
                
            # 2. Locate Non-Component Descendants: Trace all children of the component links. If a child link is not a member of the tree component, it is flagged as a structural orphan.
            
            # 3. Generate the Kinematic Proxy Rerouting: Create an explicit structural redirect map. Every structural orphan must have its joint parent updated to point directly to the component’s attachment_root (or an abstracted proxy link representing it).
            
        elif(component_type == "chain"):
            print("Abstracting Chain Components")
            # 1. Identify the Segment Ports: Track the segment's Inlet (the joint connecting the segment to the rest of the body) and its Outlet(s) (the joint(s) connecting the end of the segment to hands, tools, or head mounts).
            
            # 2. Accumulate Rigid Transformations: Programmatically multiply the static transformation matrices ($T$) of all internal fixed joints along that segment to maintain spatial accuracy.
            
            # 3. Reroute Outlets to Inlets: Sever the internal segment links, but heal the graph by connecting the Outlets directly to the Inlet's parent, updating the joint transforms with the accumulated spatial offsets.
        
    def abstract_sensor_component(self, ):
        # each sensor group must have a reference link (optical frame, ...)
        return
    
    def abstract_end_effector_component(self, ):
        # each end_effector must have a reference link, 
        return
    
    def abstract_locomotion_component(self, ):
        # each locomotion component must have a reference link (e.g., "base_link")
        return
    
    def absract_body_part_component(self, ):
        # each manipulation component must have a reference link (e.g., "shoulder_link") which ends with the end_effector link
        return
    
    def abstract_tree_components_v3(self, final_trees: dict, keep_abstracted_links: bool = False) -> dict:

        removed_topology_mask = {
            "nodes": set(),       # Every link ID removed from the active URDF
            "boundaries": {}      # Mapping of group_id -> final re-anchor point name
        }
        
        processed_count = 0

        for gid, tree_comp in final_trees.items():
            comp_type = tree_comp.type
            mount_name = tree_comp.attachment_root
            full_group_membership = set(tree_comp.all_links)

            # 1. Validation Guard
            if mount_name not in self.links:
                print(f"[SKIP] Could not find designated attachment root '{mount_name}' in all links for {gid}")
                continue
                
            mount_link = self.links[mount_name]
            processed_count += 1
            
            # Track all internal links for deletion later (excluding the root mount itself)
            for link_name in full_group_membership:
                if link_name in self.links and link_name != mount_name:
                    removed_topology_mask["nodes"].add(link_name)

            # 2. Boundary Scan: Find any child link hanging off a component node that is OUTSIDE the group
            structural_orphans: List[URDFLink] = []
            for comp_link_name in full_group_membership:
                if comp_link_name not in self.links:
                    continue
                
                comp_link = self.links[comp_link_name]
                for child in comp_link.children:
                    if child.name not in full_group_membership:
                        structural_orphans.append(child)
                        
            if(structural_orphans):
                print(f"[FOUND ORPHANS] Located orphan links:"),
                for orphan in structural_orphans:
                    print(orphan.name)

            # 3. Conditional Destination Allocation => not necessary to create this proxy link
            # Create the proxy ONLY if requested AND structural orphans actually exist
            if keep_abstracted_links and structural_orphans:
                if(tree_comp.semantic_name != ""):
                    proxy_link_name = tree_comp.semantic_name
                    proxy_link = URDFLink(proxy_link_name)
                else:
                    proxy_link_name = f"{comp_type}_{gid.split('_')[-1]}_proxy"
                    proxy_link = URDFLink(proxy_link_name)
                
                # Hook the proxy up to the physical mount point
                proxy_link.parent = mount_link
                mount_link.children.append(proxy_link)
                
                # Register the proxy link directly into the live master registry
                self.links[proxy_link_name] = proxy_link
                destination_link = proxy_link
                print(f"[PROXY CREATED] Component {gid} has {len(structural_orphans)} orphan(s). Created proxy: '{proxy_link_name}'")
            else:
                # Default fallback: Flatten straight to the original mounting frame
                destination_link = mount_link

            removed_topology_mask["boundaries"][gid] = destination_link.name

            # 4. Orphan Rerouting Pass
            for orphan in structural_orphans:
                old_parent = orphan.parent
                if old_parent and orphan in old_parent.children:
                    old_parent.children.remove(orphan)
                
                # Graft the orphan onto our designated destination link (proxy or mount)
                orphan.parent = destination_link
                destination_link.children.append(orphan)
                print(f"[REROUTE] Saved structural orphan '{orphan.name}' -> re-anchored to '{destination_link.name}'")

        # # 5. Graph Purge pass
        # for dead_node_name in removed_topology_mask["nodes"]:
        #     if dead_node_name in self.links:
        #         dead_node = self.links[dead_node_name]
        #         if dead_node.parent and dead_node in dead_node.parent.children:
        #             dead_node.parent.children.remove(dead_node)
        #         del self.links[dead_node_name]
                
        # 5. Graph Purge Pass (With Root Mount Protection)
        for dead_node_name in removed_topology_mask["nodes"]:
            if dead_node_name in final_trees and dead_node_name == final_trees[gid].attachment_root:
                print(f"[PROTECTED] Skipping deletion of attachment root node: '{dead_node_name}'")
                continue
                
            if dead_node_name in self.links:
                dead_node = self.links[dead_node_name]
                
                # Safely disconnect the dead node from its parent
                if dead_node.parent and dead_node in dead_node.parent.children:
                    dead_node.parent.children.remove(dead_node)
                    
                # Evict cleanly from master registry
                del self.links[dead_node_name]

        print(f"\n[SUCCESS] Abstracted {processed_count} tree components. Total links purged: {len(removed_topology_mask['nodes'])}")
        return removed_topology_mask

    def abstract_tree_components_v4(self, final_trees: dict) -> dict:
        removed_topology_mask = {
            "nodes": set(),       # Every link ID marked for deletion
            "boundaries": {}      # Mapping of group_id -> final re-anchor mount point name
        }
        
        # Pre-compute all attachment roots across all groups to protect them from deletion
        all_attachment_roots = {tree.attachment_root for tree in final_trees.values()}

        for gid, tree_comp in final_trees.items():
            mount_name = tree_comp.attachment_root
            full_group_membership = set(tree_comp.all_links)

            if mount_name not in self.links:
                print(f"[SKIP] Designated attachment root '{mount_name}' not found in links for {gid}")
                continue
                
            mount_link = self.links[mount_name]
            removed_topology_mask["boundaries"][gid] = mount_name
            
            # Mark all internal links for deletion (excluding the root mount itself)
            for link_name in full_group_membership:
                if link_name in self.links and link_name != mount_name:
                    removed_topology_mask["nodes"].add(link_name)

            # 1. Boundary Scan & Reroute: Find and save any child link outside the group
            for comp_link_name in full_group_membership:
                if comp_link_name not in self.links:
                    continue
                
                comp_link = self.links[comp_link_name]
                
                # Use a static list slice to safely mutate children during iteration
                for child in list(comp_link.children):
                    if child.name not in full_group_membership:
                        print(f"[WARNING] Structural orphan discovered: '{child.name}' hanging off deleted component node '{comp_link_name}'")
                        
                        # Sever connection from the dying internal node
                        comp_link.children.remove(child)
                        
                        # Graft straight onto the physical base mounting anchor
                        child.parent = mount_link
                        mount_link.children.append(child)
                        print(f"[REROUTE] Rerouted orphan '{child.name}' -> re-anchored to root mount '{mount_name}'")

        # 2. Graph Purge Pass: Physical eviction from the URDFTree structure
        purged_count = 0
        for dead_node_name in list(removed_topology_mask["nodes"]):
            if dead_node_name in all_attachment_roots:
                continue
                
            if dead_node_name in self.links:
                dead_node = self.links[dead_node_name]
                
                # Sever from parent node
                if dead_node.parent and dead_node in dead_node.parent.children:
                    dead_node.parent.children.remove(dead_node)
                    
                # Free children reference arrays to prevent lingering garbage collection tracks
                dead_node.children.clear()
                    
                # Evict from master memory registry map
                del self.links[dead_node_name]
                purged_count += 1

        print(f"\n[SUCCESS] Tree abstraction complete. Total internal links purged from structure: {purged_count}")
        return removed_topology_mask

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
            # if link.joint_to_parent_type in ("revolute", "prismatic"):
            #     ET.SubElement(joint_elem,"limit",lower="0.0", upper="0.0", effort="0.0",velocity="0.0",)
            # elif link.joint_to_parent_type == "continuous":
            #     ET.SubElement(joint_elem, "limit", effort="0.0", velocity="0.0")
                
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
           
    def build_link_element(self, link: URDFLink) -> Dict:
        """Build comprehensive context about the frame for analysis"""
        if(link.parent):
            return {
                'parent': link.parent.name,
                'joint': {"name": link.joint_name,  "type":link.joint_to_parent_type},
                'child': link.name
            }
        else:
            return {}
    
    def build_textual_link_element(self, link: URDFLink) -> str:
        # ex:  parent_link --[joint_name:joint_type]--> child_link
        elem = ""
        if(link.parent != None):
            joint = f" --[{link.joint_name}:{link.joint_to_parent_type}]--> "
            elem = link.parent.name + joint + link.name
        return elem
    
    def build_csv_hierarchy_from_leaf(self, link: URDFLink) -> str:
        # ex:  ["parent_link", "joint_name", "joint_type", "child_link"]
        elem = ""
        if(link.parent != None):
            elem = [link.parent.name, link.joint_name, link.joint_to_parent_type, link.name]
        
        return elem
    
    def build_frame_context_from_link(self, link: URDFLink, frames: List) -> List:
        """Build comprehensive context from the initial frame and its children for analysis"""
        
        if frames is None:
            frames = []

        if link is None:
            return frames

        frames.append(self.build_textual_link_element(link))

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

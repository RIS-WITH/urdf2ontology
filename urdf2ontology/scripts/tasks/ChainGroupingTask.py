from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Any, Set
import json

class BodyPartSegmentGroupingTask(VotingTask, URDFLinkTask):
    """
    Groups distinct sub-chains (dummy frames, visual links, collision geometry, 
    and structural segments) into unified macro body-part entities (e.g. head, left_arm, torso).
    
    Returns: List[List[List[str]]]
             -> List of Clustered Macro Entities (e.g., [Head Entity, Arm Entity])
               -> List of Segments in Entity ([dummyhead_chain, vis1_chain, vis2_chain])
                 -> List of URDF link strings in each segment
    """
    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.keyword_definition = ""
        self.initial_chains: Dict[str, List[str]] = {} # Maps segment_key -> List[str] of URDF links
        self.mode = mode

        self.json_format = """
{
    "groups": {
        "group_head": {
            "segments": ["chain_key_a", "chain_key_b", "chain_key_c"],
            "justification": string
        }
    }
}"""
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "object",
                    "description": f"A dictionary mapping body part group names to arrays of input segment keys that form the same macro {self.keyword} entity.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "justification": {"type": "string"}
                        },
                        "required": ["segments", "justification"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["groups"],
            "additionalProperties": False
        }

    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format

    def system_prompt(self) -> str:
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task: Cluster the provided target {self.keyword} segment keys into complete macro body-part entities (e.g., head, left_arm, right_arm, torso, left_leg).

Core Grouping Rules:
1. Merge Sub-assemblies of the Same Region: Combine dummy frames, visual links, collision meshes, sensors, and structural chains that belong to the SAME physical body region into ONE single group.
2. Cross-Branch Assembly: Segments DO NOT need to be directly connected or on the same linear serial chain. Parallel sub-chains attached to the same area or parent hub belong to the same macro group.
3. Left/Right Isolation: Keep distinct physical sides separate. Never merge left and right counterparts.
4. Key Matching Rule: In your output JSON, populate the "segments" array using ONLY the exact input dictionary keys (e.g., "chain_0", "chain_1"). Do NOT output raw internal link strings.
5. Every single input key MUST be assigned to exactly one group.

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
"""

    def user_prompt(self, user_input: dict) -> str:
        self.initial_chains = {}
        data_input = {}
        
        for segment_key, identified_comp in user_input.items():
            # Store underlying URDF link strings
            links = identified_comp.link_chain if hasattr(identified_comp, 'link_chain') else [segment_key]
            self.initial_chains[segment_key] = links
            
            data_input[segment_key] = {
                "contained_links": links,
                "semantic_type": getattr(identified_comp, 'type', ''), 
                "attachment_root": getattr(identified_comp, 'attachment_root', ''), 
                "kinematic_chain": identified_comp.get_kinematic_chain(self.mode) if hasattr(identified_comp, 'get_kinematic_chain') else []
            }
         
        chains = json.dumps(data_input, indent=2)
         
        return f"""
Below are the target {self.keyword} sub-chains requiring body-part entity clustering:

Input data:
{chains}""" 

    def interpret(self, output: dict) -> List[List[str]]:
        valid_results: List[List[str]] = []
        
        if not isinstance(output, dict):
            return valid_results

        groups_dict = output.get("groups", {})
        if not isinstance(groups_dict, dict):
            return valid_results

        # Support both segment keys ('chain_0') AND individual link strings ('dummyhead')
        valid_segment_keys = set(self.initial_chains.keys())
        link_to_segment_key = {
            link: seg_key 
            for seg_key, link_list in self.initial_chains.items() 
            for link in link_list
        }

        for group_id, group_data in groups_dict.items():
            if not isinstance(group_data, dict):
                continue

            grouped_items = group_data.get("segments", group_data.get("links", []))
            if not isinstance(grouped_items, list) or len(grouped_items) == 0:
                continue

            resolved_keys = []
            invalid_items = []

            for item in grouped_items:
                if item in valid_segment_keys:
                    resolved_keys.append(item)
                elif item in link_to_segment_key:
                    # Automatically map raw link name back to parent segment key
                    resolved_keys.append(link_to_segment_key[item])
                else:
                    invalid_items.append(item)

            if len(invalid_items) == 0 and len(resolved_keys) > 0:
                # Deduplicate keys while preserving order
                valid_results.append(list(dict.fromkeys(resolved_keys)))
            else:
                print(f"[CLUSTER ERROR] Dropping group '{group_id}' due to unresolvable items: {invalid_items}")

        print(f"Interpreted {len(valid_results)} validated macro body-part allocations.")
        return valid_results

    def aggregate(self, generations: List[List[List[str]]]) -> List[List[List[str]]]:
        pair_counts = defaultdict(int)
        all_segments: Set[str] = set()
        num_generations = len(generations)

        # 1. Track co-occurrences of segment keys
        for generation in generations:
            seen_pairs = set()
            for grouped_segments in generation:
                all_segments.update(grouped_segments)
                for a, b in combinations(sorted(grouped_segments), 2):
                    seen_pairs.add((a, b))

            for pair in seen_pairs:
                pair_counts[pair] += 1

        # 2. Build consensus adjacency graph
        adjacency = defaultdict(set)
        majority_threshold = (num_generations // 2) + 1 if num_generations > 1 else 1

        for (a, b), count in pair_counts.items():
            if count >= majority_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # 3. Extract connected entity clusters
        visited: Set[str] = set()
        final_entity_clusters: List[List[List[str]]] = []

        for segment_key in sorted(all_segments):
            if segment_key not in visited:
                stack = [segment_key]
                segment_cluster = set()

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        segment_cluster.add(node)
                        stack.extend(list(adjacency[node] - visited))

                # Map each segment key back to its list of URDF links
                entity_segments: List[List[str]] = [
                    self.initial_chains[seg] 
                    for seg in sorted(list(segment_cluster)) 
                    if seg in self.initial_chains
                ]
                
                final_entity_clusters.append(entity_segments)

        print(f"Validated {len(final_entity_clusters)} finalized macro body-part entities.")
        return final_entity_clusters
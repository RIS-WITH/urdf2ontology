from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Any, Set
import json

class ChainLinkGroupingTask(VotingTask, URDFLinkTask):
    """
    Task designed to group serial/sequential link segments into contiguous 
    chain-like component assemblies. 
    
    Returns: List[List[List[str]]] 
             -> List of Clustered Entities
               -> List of Segments in Entity
                 -> List of URDF Links in Segment
    """
    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.keyword_definition = ""
        self.initial_chains: Dict[str, List[str]] = {} # Maps segment_id -> List[str] of URDF links
        self.mode = mode

        self.json_format = """
{
    "groups": {
        "group_0": {
            "segments": ["segment_key_a", "segment_key_b"],
            "justification": string
        }
    }
}"""
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "object",
                    "description": f"A dictionary mapping arbitrary group IDs to arrays of segment keys forming a continuous {self.keyword} chain.",
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

        task_domain = "distinct macro body parts or physical device assemblies" if self.keyword in ["body_part", "locomotion", "manipulator"] else "distinct functional device systems"
        
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task: Cluster the provided target {self.keyword} segments into {task_domain}.

Context:
The input segments have ALREADY been chunked into localized link segments. Your goal is to group segments together if they belong to the SAME overarching body part entity, EVEN IF THEY ARE NOT CONTIGUOUS or originate from DIFFERENT kinematic chain ancestors.

Grouping Logic:
1. One Group = One complete macro body part or device assembly unit.
2. Cross-Branch & Non-Contiguous Assembly: Segments DO NOT need to be directly connected or on the same linear serial chain. If segments share a functional or anatomical role within the same macro body part (even if they branch off different root ancestors), they MUST be grouped together.
3. Left/Right & Symmetry Disambiguation: Keep distinct physical entities separate. Never merge left and right counterparts, and do not merge distinct macro regions.
4. Absolute Requirement: Every single target element key provided in the input must be assigned to exactly one group inside your response.

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining why the selected segments form the same body part entity.
"""

    def user_prompt(self, user_input: dict) -> str:
        self.initial_chains = {}
        data_input = {}
        
        for segment_key, identified_comp in user_input.items():
            # Store the underlying list of links for each segment ID
            self.initial_chains[segment_key] = identified_comp.link_chain if hasattr(identified_comp, 'link_chain') else [segment_key]
            
            data_input[segment_key] = {
                "semantic_type": getattr(identified_comp, 'type', ''), 
                "attachment_root": getattr(identified_comp, 'attachment_root', ''), 
                "kinematic_chain": identified_comp.get_kinematic_chain(self.mode) if hasattr(identified_comp, 'get_kinematic_chain') else []
            }
         
        chains = json.dumps(data_input, indent=2)
         
        return f"""
Below are the identified target {self.keyword} chain segments requiring grouping evaluation:

Input data: {chains}""" 

    # def interpret(self, output: dict) -> List[List[str]]:
    #     """
    #     Validates LLM segment grouping predictions against input segment keys.
    #     Returns: List[List[str]] (groups of segment keys) for consensus aggregation.
    #     """
    #     valid_results: List[List[str]] = []
        
    #     if not isinstance(output, dict):
    #         return valid_results

    #     groups_dict = output.get("groups", {})
    #     if not isinstance(groups_dict, dict):
    #         print(f"[PARSE ERROR] 'groups' field missing or is not a JSON object dictionary.")
    #         return valid_results

    #     valid_segment_keys = set(self.initial_chains.keys())

    #     for group_id, group_data in groups_dict.items():
    #         if not isinstance(group_data, dict):
    #             continue

    #         # Support both 'segments' and 'links' key fallback
    #         grouped_segments = group_data.get("segments", group_data.get("links", []))

    #         if not isinstance(grouped_segments, list) or len(grouped_segments) == 0:
    #             continue
            
    #         # Ground-truth validation against input keys
    #         if all(seg in valid_segment_keys for seg in grouped_segments):
    #             valid_results.append(list(grouped_segments))
    #         else:
    #             invalid_nodes = [s for s in grouped_segments if s not in valid_segment_keys]
    #             print(f"[CLUSTER ERROR] Dropping group '{group_id}' due to out-of-bounds segments: {invalid_nodes}")

    #     print(f"Interpreted {len(valid_results)} validated constituent chain segment allocations.")
    #     return valid_results
    def interpret(self, output: dict) -> List[List[str]]:
        valid_results: List[List[str]] = []
        
        if not isinstance(output, dict):
            return valid_results

        groups_dict = output.get("groups", {})
        if not isinstance(groups_dict, dict):
            print(f"[PARSE ERROR] 'groups' field missing or is not a JSON object dictionary.")
            return valid_results

        # 1. Build a reverse mapping from link_name -> segment_key
        link_to_segment_key = {}
        for seg_key, link_list in self.initial_chains.items():
            for link in link_list:
                link_to_segment_key[link] = seg_key

        valid_segment_keys = set(self.initial_chains.keys())

        for group_id, group_data in groups_dict.items():
            if not isinstance(group_data, dict):
                continue

            grouped_items = group_data.get("segments", group_data.get("links", []))

            if not isinstance(grouped_items, list) or len(grouped_items) == 0:
                continue

            resolved_segment_keys = []
            invalid_nodes = []

            for item in grouped_items:
                if item in valid_segment_keys:
                    resolved_segment_keys.append(item)
                elif item in link_to_segment_key:
                    # Convert raw link name back to its segment key ('chain_0')
                    resolved_segment_keys.append(link_to_segment_key[item])
                else:
                    invalid_nodes.append(item)

            if len(invalid_nodes) == 0 and len(resolved_segment_keys) > 0:
                # Deduplicate segment keys in case multiple links from same chain were listed
                valid_results.append(list(dict.fromkeys(resolved_segment_keys)))
            else:
                print(f"[CLUSTER ERROR] Dropping group '{group_id}' due to out-of-bounds segments: {invalid_nodes}")

        print(f"Interpreted {len(valid_results)} validated body-part segment allocations.")
        return valid_results

    def aggregate(self, generations: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Aggregates consensus across generations using majority graph voting on segment keys,
        then maps each segment key back to its list of links.
        
        Returns: List[List[List[str]]]
        """
        pair_counts = defaultdict(int)
        all_segments: Set[str] = set()
        num_generations = len(generations)

        # --- Step 1: Track Segment Co-occurrences ---
        for generation in generations:
            seen_pairs = set()
            for grouped_segments in generation:
                all_segments.update(grouped_segments)
                for a, b in combinations(sorted(grouped_segments), 2):
                    seen_pairs.add((a, b))

            for pair in seen_pairs:
                pair_counts[pair] += 1

        # --- Step 2: Majority Graph Layout ---
        adjacency = defaultdict(set)
        majority_threshold = (num_generations // 2) + 1 if num_generations > 1 else 1

        for (a, b), count in pair_counts.items():
            if count >= majority_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # --- Step 3: Connected Components (BFS/DFS Engine) ---
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

        print(f"Validated {len(final_entity_clusters)} finalized {self.keyword} chain physical entities.")
        return final_entity_clusters
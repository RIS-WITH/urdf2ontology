from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Any, Set
import json

class LinkGroupingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "groups": {
        "group_0": {
            "links": ["leaf_link_a", "leaf_link_b"],
            "justification": string
            }
    }
}"""
        
        self.structured_format = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "object",
            "description": f"A dictionary mapping arbitrary tracking group IDs to group objects.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "links": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of physical leaf link names."
                    },
                    "justification": {
                        "type": "string",
                        "description": "Reason for grouping these links together."
                    }
                },
                "required": ["links", "justification"],
                "additionalProperties": False
            }
        }
    },
    "required": ["groups"],
    "additionalProperties": False
}
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format

class ComponentLinkGroupingTask(LinkGroupingTask):
    def __init__(self, nb_runs : int, keyword : str, mode: str):
        super().__init__(nb_runs, keyword)
        self.keyword_definition = ""
        self.initial_chains = {}
        self.mode = mode

    def system_prompt(self) -> str:
        task_domain = "distinct functional groups" if self.keyword in ["body_part", "locomotion"] else "distinct physical device assemblies"
    
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task: Cluster the provided target leaf {self.keyword} links into {task_domain} based on their kinematic subchains, attachment roots, and semantic device identities.
As each clustered group will be used to create a typed individual representing a single {self.keyword} unit, keep distinct physical devices separate, UNLESS they form a unified, coordinated functional system (such as a multi-wheel mobile base).

Grouping Logic:
1. Shared Device Assembly (Merge): Merge elements if they form parts/sub-frames of the same distinct device assembly (e.g., multi-sensor modules, optical/device frame pairs, or multi-branch mechanisms sharing a dedicated, immediate local mounting base).
2. Special Rule for Locomotion Systems:
   - If keyword is 'locomotion', do NOT separate individual parts into isolated groups if they collectively form a single mobility unit.
   - Merge all locomotion subchains sharing a common structural chassis into a single composite locomotion group representing the complete locomotion subsystem.
3. Functional & Semantic Separation (Do NOT Merge Sensors/Body Parts): Distinct physical devices or distinct sensor modalities (e.g., separate cameras, independent manipulators) that happen to be attached to the same broad structural parent MUST remain in separate groups.
4. Conservatism Constraint: For non-locomotion items, if structural connection or functional kinship is ambiguous, do NOT merge. Keep them in separate groups.
5. Absolute Requirement: Every single target element provided in the input must be assigned to exactly one group inside your response.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting.
"""
    def user_prompt(self, user_input: dict) -> str:
        self.initial_chains = {}
        data_input = {}
        
        for link, identified_comp in user_input.items():
            self.initial_chains[link] = identified_comp.link_chain # Store only the base link_chain for verification passes
            data_input[link] = {
                "semantic_type": identified_comp.type, 
                "attachment_root": identified_comp.attachment_root, 
                "kinematic_chain": identified_comp.get_kinematic_chain(self.mode)
            }
         
        chains = json.dumps(data_input, indent=2)
         
        new_message = f"""
Below are the identified target {self.keyword} elements requiring grouping evaluation:

Input data: {chains}""" 

        return new_message
        
    def interpret(self, output: dict) -> List[List[str]]:
        valid_results: List[List[str]] = []
        
        if not isinstance(output, dict):
            return valid_results

        groups_dict = output.get("groups", {})
        if not isinstance(groups_dict, dict):
            print(f"[PARSE ERROR] 'groups' field missing or is not a JSON object dictionary.")
            return valid_results

        for group_id, group_data in groups_dict.items():
            if not isinstance(group_data, dict):
                continue

            grouped_links = group_data.get("links", [])
            justification = group_data.get("justification", "")

            if not isinstance(grouped_links, list) or len(grouped_links) == 0:
                continue
            
            if all(link in self.initial_chains for link in grouped_links):
                valid_results.append(list(grouped_links))
            else:
                invalid_nodes = [l for l in grouped_links if l not in self.initial_chains]
                print(f"[CLUSTER ERROR] Dropping group '{group_id}' due to out-of-bounds target elements: {invalid_nodes}")

        print(f"Interpreted {len(valid_results)} validated constituent group allocations.")
        return valid_results
    
    def aggregate(self, generations: List[List[List[str]]]) -> List[List[str]]:
        pair_counts = defaultdict(int)
        all_links: Set[str] = set()
        num_generations = len(generations)

        # --- Step 1: Track Co-occurrences ---
        for generation in generations:
            seen_pairs = set()
            for grouped_links in generation:
                # Track all links to catch single-element groups safely
                all_links.update(grouped_links)
                
                # Register link-to-link pairs
                for a, b in combinations(sorted(grouped_links), 2):
                    seen_pairs.add((a, b))

            for pair in seen_pairs:
                pair_counts[pair] += 1

        # --- Step 2: Extract Majority Graph Layout ---
        adjacency = defaultdict(set)
        
        # Rigorous Strict Majority Rule calculation logic
        majority_threshold = (num_generations // 2) + 1 if num_generations > 1 else 1

        for (a, b), count in pair_counts.items():
            if count >= majority_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # --- Step 3: Run Connected Components (BFS/DFS Engine) ---
        visited: Set[str] = set()
        final_validated_groups: List[List[str]] = []

        for link in sorted(all_links):
            if link not in visited:
                stack = [link]
                component_cluster = set()

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component_cluster.add(node)
                        # Append any connected peer links that cleared consensus metrics
                        stack.extend(list(adjacency[node] - visited))

                final_validated_groups.append(sorted(list(component_cluster)))

        print(f"Validated {len(final_validated_groups)} finalized {self.keyword} physical groups.")
        return final_validated_groups
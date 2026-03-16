from tasks.VotingTask import VotingTask

from itertools import combinations
from collections import defaultdict
from typing import List, Tuple
import json

class LinkGroupingTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = f"""
        {{
            "grouped_{self.keyword}s": [
                {{
                    "grouped_leaf_links": ["string", "..."]
                }}
            ]
        }}
        """
        
        self.structured_format = {
            "type": "object",
            "properties": {
                f"grouped_{self.keyword}s": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "grouped_leaf_links": {
                        "type": "array",
                        "items": {"type": "string"}
                        }
                    },
                    "required": ["grouped_leaf_links"],
                    "additionalProperties": False
                }
                }
            },
            "required": [f"grouped_{self.keyword}s"],
            "additionalProperties": False
            }

class ComponentLinkGroupingTask(LinkGroupingTask):
    def __init__(self, nb_runs : int, keyword : str):
        super().__init__(nb_runs, keyword)
        self.keyword_definition = ""
        self.initial_chains = {}
        
    def system_prompt(self) -> str:

        compacted_prompt = f""" 
Task: Group {self.keyword} leaf links into distinct physical device groups.

Grouping Logic:
1. One Group = One physical device (e.g., a single {self.keyword} assembly unit, consisting of one or several leaf links).
2. Evidence: Use the provided kinematic subchains to identify intrinsic structural cohesion between links.
3. Strategy: Be conservative. If kinship is ambiguous, maintain separate groups.

Constraint: Every input leaf link must be assigned to exactly one group.

Format:
Respond strictly in JSON:
{self.json_format}
"""
        return compacted_prompt
       
    def user_prompt(self, user_input) -> str:
        
        self.initial_chains = user_input
        chains = json.dumps(user_input, indent=2)
        
        new_message = f"""
        Below are the identified {self.keyword} leaf links and their intrinsic subchains.
        
        Input data:
        {chains}
        """
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[List[str], str]]:

        valid_results = []
        grouped_sensors_chains = output.get(f"grouped_{self.keyword}s")
        
        if not isinstance(grouped_sensors_chains, list):
            return valid_results
        
        for grouped_sensor_item in grouped_sensors_chains:
            if not isinstance(grouped_sensor_item, dict):
                continue

            leaf_links = grouped_sensor_item.get("grouped_leaf_links", [])
            
            if(leaf_links == []):
                print(f"discarded grouped {self.keyword} because of empty links")
                continue
            
            if not all(link in self.initial_chains.keys() for link in leaf_links):
                print(f"discarded grouped {self.keyword} because of non-existing link as one of the leaves: {leaf_links}")
                continue
        
            valid_results.append((leaf_links))
        
        print("Interpreted", len(valid_results), f"{self.keyword} groups.")
         
        return valid_results
    
    def aggregate(self, generations: List[List[Tuple[List[str], str]]]):
        pair_counts = defaultdict(int)
        all_links = set()
        num_generations = len(generations)

        # --- Step 1: count co-occurrences ---
        for generation in generations:
            seen_pairs = set()

            for grouped_links in generation:
                all_links.update(grouped_links)
                for a, b in combinations(sorted(grouped_links), 2):
                    seen_pairs.add((a, b))

            for pair in seen_pairs:
                pair_counts[pair] += 1

        # --- Step 2: build majority graph ---
        adjacency = defaultdict(set)
        majority_threshold = int(num_generations / 2)

        for (a, b), count in pair_counts.items():
            if count > majority_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # --- Step 3: find connected components ---
        visited = set()
        final_groups = []

        for link in all_links:
            if link not in visited:
                stack = [link]
                component = set()

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.add(node)
                        stack.extend(adjacency[node] - visited)

                final_groups.append(component)

        print("Validated", len(final_groups), f"{self.keyword} groups")
        return final_groups

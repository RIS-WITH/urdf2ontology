from tasks.VotingTask import VotingTask

from itertools import combinations
from collections import defaultdict, Counter
from typing import List, Tuple
import json

class SensorLinkGroupingTask(VotingTask):
    def __init__(self, nb_runs):
        self.nb_runs = nb_runs
        self.initial_chains = {}
        self.initial_chains_joined = {}
        self.keyword = "sensor"
        
        # self.json_format = """
        # {
        #     "grouped_sensors" : [
        #         {"links": [string, string, ...], "subchains": [[string, ...], [string, ...]]},
        #     ]
        # }
        # """
        
        # self.json_format = """
        # {
        #     """ + f"{self.keyword}_sensors" + """ : [
        #         {"grouped_links": [string, string, ...]},
        #     ]
        # }
        # """
        
        self.json_format = """
        {
            "grouped_sensors" : [
                {"grouped_leaf_links": [string, ...], "common_parent": string},
            ]
        }
        """
        
        self.structured_format = {
            "type": "object",
            "properties": {
                f"{self.keyword}_sensors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "grouped_links": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                    },
                    "required": ["grouped_links"],
                    "additionalProperties": False
                }
                }
            },
            "required": [f"grouped_{self.keyword}"],
            "additionalProperties": False
            }
        
    def system_prompt(self) -> str:
        
        compacted_task_prompt = """
        Task:
        You are given sensor links along with their trimmed sensor subchains. Group them into sensor components, where each component corresponds to one physical, hot-swappable sensor device. A sensor component may contain multiple links (so subchains) if they represent different internal frames or functional parts of the same physical sensor.

        Grouping Rules:
        Group links into the same component if their subchains:
        -Share intrinsic sensor links, or
        -Represent different internal frames or functional parts of the same device, or
        -Attach via the same mounting point and clearly describe one sensor.
        Do not group links from different physical devices.

        Input:
        A list of sensor links + trimmed sensor subchains (each an ordered list of intrinsic sensor link names).

        Output:
        Return a list of sensor components based on the links. For each component, produce:
        - grouped_leaf_links: Leaf link names belonging to the component.
        - common_parent: Link that matches all internal frames of a sensor subgroup, discarding upper structural links.

        Constraints:
        Every input link must appear in exactly one component.
        Do not introduce new links.
        Prefer conservative grouping when uncertain.
        
        Respond only with the following JSON format:""" + self.json_format
        
        # simple_prompt = """
        # Task:
        # You are given sensor attachment subchains extracted from a URDF.
        # Each subchain is an ordered list of links (parent -> child) starting at a robot-to-sensor attachment root and ending at a sensor-related leaf link. All subchains in each element of the list in input share the same attachment root.

        # Your mission is to group subchains that correspond to the same physical, hot-swappable sensor device. You must group them into a subgroup if they represent different internal frames or functional parts of the same device. 
        # Do not group subchains that represent different physical devices.

        # Input:
        # A list of leaf-link + subchain pairs, per previously identified monting point.

        # Output:
        # For each sensor component:
        # -subgroup_links: leaf link names in the subgroup.
        # -subgroup_mounting_point: the deepest link shared by all subchains in the subgroup, which matches all internal frames of a sensor subgroup.

        # Constraints
        # -Each link + subchain must belong to exactly one group.
        
        # Respond only with the following JSON format:
        # """ + self.json_format
        
        return compacted_task_prompt
       
    def user_prompt(self, user_input) -> str:
        
        self.initial_chains = user_input
        
        user_prompt = f"Here is the list of {self.keyword} component chains that could be merged together: " + json.dumps(user_input, indent=2)
        
        return user_prompt
    
    
    # version without chains 
    def interpret(self, output: dict) -> List[Tuple[List[str], str]]:
        valid_results = []
        # == each vote is of the form [links=["leaf_1, leaf2"], chains= [["parent_1", leaf_1], ["parent_2", leaf_2] ]
        # print("======= Interpreting =======")
       
        valid_results = []
        grouped_sensors_chains = output.get(f"grouped_{self.keyword}s")
        
        if not isinstance(grouped_sensors_chains, list):
            return valid_results
        
        for grouped_sensor_item in grouped_sensors_chains:
            # Defensive parsing
            if not isinstance(grouped_sensor_item, dict):
                continue

            leaf_links = grouped_sensor_item.get("grouped_leaf_links", [])
            common_parent_link = grouped_sensor_item.get("common_parent", "")
            
            if(leaf_links == [] or common_parent_link == ""):
                print(f"discarded grouped {self.keyword} because of non-existing link as one of the leaves: {leaf_links}")
                continue
            
            # Validate that all leaf links exist
            if not all(link in self.initial_chains for link in leaf_links):
                print(f"discarded grouped {self.keyword} because of non-existing link as one of the leaves: {leaf_links}")
                continue
            
            if(not all(common_parent_link in self.initial_chains[link] for link in leaf_links)):
                print(f"discarded grouped {self.keyword} because of non-existing common parent: {leaf_links}")
                continue
            
            valid_results.append((leaf_links, common_parent_link))
        
        print("Interpreted", len(valid_results), f"{self.keyword} groups.")
         
        return valid_results
    
    def aggregate(self, generations: List[List[Tuple[List[str], str]]]):
        pair_counts = defaultdict(int)
        all_links = set()
        num_generations = len(generations)

        # --- Step 1: count co-occurrences ---
        for generation in generations:
            seen_pairs = set()

            for grouped_links, _ in generation:
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

        # --- Step 4: majority vote for parent ---
        result = []

        for group in final_groups:
            parent_counter = Counter()

            for generation in generations:
                for grouped_links, parent in generation:
                    if set(grouped_links) == group:
                        parent_counter[parent] += 1

            if parent_counter:
                majority_parent = parent_counter.most_common(1)[0][0]
            else:
                majority_parent = None  # no exact match found

            result.append((list(group), majority_parent))

        print("Validated", len(result), f"{self.keyword} groups")
        return result

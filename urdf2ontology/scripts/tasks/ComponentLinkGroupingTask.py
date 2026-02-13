from tasks.VotingTask import VotingTask

from itertools import combinations
from collections import defaultdict, Counter
from typing import List, Tuple
import json

class LinkGroupingTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
        {
            "grouped_""" + f"{self.keyword}s" + """ : [
                {"grouped_leaf_links": [string, ...], "common_parent": string},
            ]
        }
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
                        },
                        "common_parent": {"type": "string"}
                    },
                    "required": ["grouped_leaf_links", "common_parent"],
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
        
        old_prompt = """
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
        
        new_prompt = f"""
        Task:
        You are given {self.keyword} leaf links along with their trimmed intrinsic subchains. Each subchain represents only the links belonging to the {self.keyword} assembly, starting from its mounting point and ending at the leaf link.

        Your task:
        Group leaf links into physical {self.keyword} components.

        Definition of a component:
        A component corresponds to exactly one physical device.
        A device may contain multiple internal frames represented as multiple leaf links.

        Grouping rules:
        Group links together if:
        - Their intrinsic subchains share a common prefix beyond the mounting point, OR
        - They represent internal frames of the same physical device, OR
        - They attach via the same mounting point AND clearly represent one device.

        Do NOT group links if:
        - They belong to distinct physical devices
        - Their subchains diverge immediately after the mounting point

        Constraints:
        - Each leaf link must appear in exactly one group.
        - Do not invent new links.
        - Prefer conservative grouping if uncertain.
        - Each group must contain at least one leaf link.

        For each group, return:
        - grouped_leaf_links
        - common_parent: the deepest shared link in the intrinsic subchains of the group

        Respond only in JSON format:
        {self.json_format}
        """
        
        return new_prompt
       
    def user_prompt(self, user_input) -> str:
        
        self.initial_chains = user_input
        chains = json.dumps(user_input, indent=2)
        
        old_message = f"Here is the list of {self.keyword} component chains that could be merged together: " + chains
        
        new_message = f"""
        You are given identified {self.keyword} leaf links along with their intrinsic subchains.
        Each entry contains:
        - A leaf link name
        - Its trimmed intrinsic subchain (ordered list of links belonging only to the {self.keyword} assembly)

        Important:
        These subchains already exclude upper structural robot links.
        They represent only links internal to the component assembly.

        Your task:
        Group leaf links that belong to the same physical {self.keyword} device.

        Constraints:
        - Every leaf link must appear in exactly one group.
        - Do not invent new links.
        - Do not omit any link.
        - Prefer conservative grouping when uncertain.
        - Groups correspond to physical devices.

        For each group:
        - grouped_leaf_links: list of leaf links in that device
        - common_parent: the deepest shared link inside the intrinsic subchains

        Input data:{chains}
        """
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[List[str], str]]:

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
                    # if set(grouped_links) == group:
                    if set(grouped_links).issubset(group):
                        parent_counter[parent] += 1

            if parent_counter:
                majority_parent = parent_counter.most_common(1)[0][0]
            else:
                majority_parent = None  # no exact match found

            result.append((list(group), majority_parent))

        print("Validated", len(result), f"{self.keyword} groups")
        return result

from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from itertools import combinations
from collections import defaultdict, Counter
from typing import List, Dict, Any, Set, Tuple
import json


class LinkGroupingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "groups": {
        "group_0": {
            "links": ["leaf_link_a", "leaf_link_b"],
            "justification": "Clear single-sentence justification explaining why these links form a physical/functional group."
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

Task: Cluster the provided target leaf {self.keyword} links into {task_domain} based on their kinematic subchains, attachment roots, and semantic device identities. As each clustered group will be used to create a typed individual representing a single {self.keyword} unit, keep distinct physical devices separate, UNLESS they form a unified, coordinated functional system.

Grouping Logic:
1. Shared Device Assembly (Merge): 
    - Merge elements if they form parts, sub-components, or reference frames of the same distinct device assembly.
2. Special Rule for Locomotion Systems: If keyword is 'locomotion', do NOT separate individual parts into isolated groups if they collectively form a single mobility modality.
3. Functional & Semantic Separation: Separate, distinct physical hardware devices MUST remain in separate groups, even if attached to the same structural parent.
4. Conservatism Constraint: Do NOT merge two completely separate physical devices together. However, sub-frames, optical frames, and reference links belonging to the same physical unit MUST be merged with that unit.
5. Absolute Requirement: Every single target element provided in the input must be assigned to exactly one group inside your response.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting.
"""
    def user_prompt(self, user_input: dict) -> str:
        # self.initial_chains = {}
        # data_input = {}
        
        # for link, identified_comp in user_input.items():
        #     self.initial_chains[link] = identified_comp.link_chain # Store only the base link_chain for verification passes
        #     data_input[link] = {
        #         "semantic_type": identified_comp.type, 
        #         "attachment_root": identified_comp.attachment_root, 
        #         "kinematic_chain": identified_comp.get_kinematic_chain(self.mode)
        #     }
         
        compact_data = {}
        for link, identified_comp in user_input.items():
            self.initial_chains[link] = identified_comp.link_chain
            
            compact_data[link] = {
                "type": identified_comp.type,
                "root": identified_comp.attachment_root,
                "chain": " -> ".join(identified_comp.get_kinematic_chain(self.mode)) # arrow-separated string instead of a long array
            }

        chains = json.dumps(compact_data, indent=2)
         
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
            if not isinstance(grouped_links, list) or len(grouped_links) == 0:
                continue
            
            # Keep valid target elements; strip out hallucinated ones
            valid_links = [l for l in grouped_links if l in self.initial_chains]
            invalid_nodes = [l for l in grouped_links if l not in self.initial_chains]

            if invalid_nodes:
                print(f"[CLUSTER WARNING] Stripped out-of-bounds elements {invalid_nodes} from group '{group_id}'")

            # Keep the group as long as at least one valid link remains
            if valid_links:
                valid_results.append(valid_links)

        print(f"Interpreted {len(valid_results)} validated constituent group allocations.")
        return valid_results
    
    # Iterative Frequent-set Extraction
    def aggregate(self, generations: List[List[List[str]]], min_votes: int = 2) -> List[List[str]]:
        """
        Extracts high-frequency exact groups and flushes their constituent 
        links from remaining runs to guarantee valid non-overlapping groups.
        
        :param generations: List of outputs from valid runs (each run is a list of link groups).
        :param min_votes: Minimum occurrences required across runs to accept a group (default: 2).
        """
        if not generations:
            return []

        # Track every link encountered to ensure 100% target coverage at the end
        all_target_links: Set[str] = set()
        for gen in generations:
            for group in gen:
                all_target_links.update(group)

        # Convert each run's groups into list of frozensets for easy set operations
        runs_groups: List[List[Tuple[str, ...]]] = []
        for gen in generations:
            run_sets = [tuple(sorted(group)) for group in gen if len(group) > 1]
            runs_groups.append(run_sets)

        # Step 1: Count occurrences of exact multi-element groups across all runs
        group_counts = Counter()
        for run in runs_groups:
            # Using set(run) per generation so a single run doesn't artificially inflate count
            for group in set(run):
                group_counts[group] += 1

        # Step 2: Sort candidates by (1) Frequency descending, (2) Group Size descending
        # sorted_candidates = sorted(
        #     group_counts.items(),
        #     key=lambda item: (item[1], len(item[0])),
        #     reverse=True
        # )
        
        # Sort by: 1. Frequency (Descending), 2. Group Size (Ascending)
        sorted_candidates = sorted(
            group_counts.items(),
            key=lambda item: (item[1], -len(item[0])),
            reverse=True
        )

        final_validated_groups: List[List[str]] = []
        claimed_links: Set[str] = set()

        # Step 3: Greedily select top groups and flush their links
        for group_tuple, count in sorted_candidates:
            if count < min_votes:
                break  # Candidates are sorted by frequency, so we can stop early

            group_set = set(group_tuple)

            # Skip if ANY link in this candidate group has already been claimed by a higher-voted group
            if group_set.intersection(claimed_links):
                continue

            # Accept group and flush its links
            final_validated_groups.append(sorted(list(group_set)))
            claimed_links.update(group_set)

        # Step 4: Cleanup / Fallback for any links not in a multi-element group
        unclaimed_links = all_target_links - claimed_links
        for single_link in sorted(unclaimed_links):
            final_validated_groups.append([single_link])

        print(f"Validated {len(final_validated_groups)} finalized {self.keyword} physical groups.")
        return final_validated_groups

    # Pairwise Graph Consensus
    # def aggregate(self, generations: List[List[List[str]]]) -> List[List[str]]:
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
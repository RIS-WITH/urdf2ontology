from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from itertools import combinations
from collections import defaultdict, Counter
from typing import List, Dict, Any, Set, Tuple
import json

class ChainGroupingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs: int, keyword: str):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "groups": {
        "group_0": {
            "chains": ["chain_0", "chain_1"],
            "justification": "Clear single-sentence justification explaining why these chains form a unified physical/functional tree assembly."
        }
    }
}"""
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "object",
                    "description": "A dictionary mapping arbitrary group IDs to merged chain group objects.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "chains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of input chain component identifiers (e.g., chain_0, chain_1)."
                            },
                            "justification": {
                                "type": "string",
                                "description": "Reason for grouping these chains together into a tree assembly."
                            }
                        },
                        "required": ["chains", "justification"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["groups"],
            "additionalProperties": False
        }
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format


class ComponentChainGroupingTask(ChainGroupingTask):
    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.keyword_definition = ""
        self.initial_chain_ids: Set[str] = set()
        self.mode = mode

    def system_prompt(self) -> str:
        task_domain = (
            "distinct macro body part entities" 
            if self.keyword in ["body_part", "locomotion"] 
            else "distinct physical device assemblies"
        )
    
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task: Cluster the provided target {self.keyword} ChainComponents into {task_domain}.

Context:
The input chains have ALREADY been chunked into localized, ordered segment sequences. Your goal is to group chains together if they belong to the SAME overarching {self.keyword} entity, EVEN IF THEY ARE NOT CONTIGUOUS or originate from DIFFERENT kinematic chain ancestors.

Grouping Rules:
1. One Group = One complete macro {self.keyword} unit or device assembly.
2. Cross-Branch & Non-Contiguous Assemblies: Chains DO NOT need to be directly connected or on the same linear serial chain. If chains share a functional or structural role within the same macro entity (e.g., a dedicated mounting housing subchain and a main manipulator branch), they MUST be grouped together.
3. Left/Right & Symmetry Disambiguation: Keep distinct physical entities separate. Never merge left and right counterparts if they represent two different physical structures, and do not merge distinct macro regions.
4. Complete Coverage Requirement: Every single target chain key provided in the input must be assigned to exactly one group in your response.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in JSON matching this schema: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining why the selected chains belong to the same macro entity.
Do not include any extra text or markdown formatting.
"""

    def user_prompt(self, user_input: dict) -> str:
        """
        user_input: Dict[str, ChainComponent] 
        Maps chain_id (e.g. 'chain_0') to its ChainComponent instance.
        """
        self.initial_chain_ids = set(user_input.keys())
        data_input = {}
        
        for chain_id, chain_comp in user_input.items():
            # Extract ordered segment information
            formatted_segments = []
            for seg in getattr(chain_comp, "segments", []):
                formatted_segments.append({
                    "segment_id": getattr(seg, "id", getattr(seg, "semantic_name", "")),
                    "type": getattr(seg, "type", "ChainSegment"),
                    "link_chain": list(getattr(seg, "link_chain", []))
                })

            data_input[chain_id] = {
                "chain_type": getattr(chain_comp, "type", self.keyword.capitalize()),
                "attachment_root": getattr(chain_comp, "attachment_root", ""),
                "assembly_base_link": getattr(chain_comp, "assembly_base_link", None),
                "ordered_segments": formatted_segments,
                "all_links": list(getattr(chain_comp, "all_links", []) or getattr(chain_comp, "link_chain", []))
            }
         
        chains_json = json.dumps(data_input, indent=2)
         
        new_message = f"""
Below are the identified target {self.keyword} ChainComponents requiring grouping evaluation:

Input data: {chains_json}""" 

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

            grouped_chains = group_data.get("chains", [])
            if not isinstance(grouped_chains, list) or len(grouped_chains) == 0:
                continue

            # Keep valid chain IDs; strip out any hallucinated IDs
            valid_chains = [c for c in grouped_chains if c in self.initial_chain_ids]
            hallucinated = [c for c in grouped_chains if c not in self.initial_chain_ids]

            if hallucinated:
                print(f"[CLUSTER WARNING] Stripped invalid chain IDs {hallucinated} from group '{group_id}'")

            if valid_chains:
                valid_results.append(valid_chains)

        print(f"Interpreted {len(valid_results)} validated constituent chain group allocations.")
        return valid_results

    # Iterative Frequent-set Extraction
    def aggregate(self, generations: List[List[List[str]]], min_votes: int = 2) -> List[List[str]]:
        """
        Extracts high-frequency exact chain groups and flushes their constituent 
        chains from remaining runs to guarantee non-overlapping groups.
        """
        if not generations:
            return []

        # Track all valid initial chain IDs to ensure 100% coverage
        all_target_chains: Set[str] = set(self.initial_chain_ids)

        # Convert each run's groups into sorted tuples
        runs_groups: List[List[Tuple[str, ...]]] = []
        for gen in generations:
            run_sets = [tuple(sorted(group)) for group in gen if len(group) > 1]
            runs_groups.append(run_sets)

        # Count frequency of exact candidate groups across runs
        group_counts = Counter()
        for run in runs_groups:
            for group in set(run):  # Dedup within a single run
                group_counts[group] += 1

        # Sort candidates by: (1) Vote count descending, (2) Group size descending
        sorted_candidates = sorted(
            group_counts.items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True
        )

        final_validated_groups: List[List[str]] = []
        claimed_chains: Set[str] = set()

        # Greedily select top groups and flush claimed chains
        for group_tuple, count in sorted_candidates:
            if count < min_votes:
                break

            group_set = set(group_tuple)

            # Skip if any chain in this group was already claimed by a higher-ranked group
            if group_set.intersection(claimed_chains):
                continue

            final_validated_groups.append(sorted(list(group_set)))
            claimed_chains.update(group_set)

        # Fallback: assign remaining unclaimed chains as singletons
        unclaimed_chains = all_target_chains - claimed_chains
        for single_chain in sorted(unclaimed_chains):
            final_validated_groups.append([single_chain])

        print(f"Validated {len(final_validated_groups)} finalized {self.keyword} chain groups.")
        return final_validated_groups
    
    # Pairwise Graph Consensus
    # def aggregate(self, generations: List[List[List[str]]]) -> List[List[str]]:
    #     pair_counts = defaultdict(int)
    #     all_chains: Set[str] = set()
    #     num_generations = len(generations)

    #     # --- Step 1: Track Co-occurrences ---
    #     for generation in generations:
    #         seen_pairs = set()
    #         for grouped_chains in generation:
    #             # Track all chains to handle single-chain groups safely
    #             all_chains.update(grouped_chains)
                
    #             # Register chain-to-chain pairs
    #             for a, b in combinations(sorted(grouped_chains), 2):
    #                 seen_pairs.add((a, b))

    #         for pair in seen_pairs:
    #             pair_counts[pair] += 1

    #     # --- Step 2: Extract Majority Graph Layout ---
    #     adjacency = defaultdict(set)
        
    #     # Strict Majority Rule calculation
    #     majority_threshold = (num_generations // 2) + 1 if num_generations > 1 else 1

    #     for (a, b), count in pair_counts.items():
    #         if count >= majority_threshold:
    #             adjacency[a].add(b)
    #             adjacency[b].add(a)

    #     # --- Step 3: Run Connected Components (BFS/DFS Engine) ---
    #     visited: Set[str] = set()
    #     final_validated_groups: List[List[str]] = []

    #     for chain_id in sorted(all_chains):
    #         if chain_id not in visited:
    #             stack = [chain_id]
    #             component_cluster = set()

    #             while stack:
    #                 node = stack.pop()
    #                 if node not in visited:
    #                     visited.add(node)
    #                     component_cluster.add(node)
    #                     # Append connected peer chains that cleared consensus metrics
    #                     stack.extend(list(adjacency[node] - visited))

    #             final_validated_groups.append(sorted(list(component_cluster)))

    #     print(f"Validated {len(final_validated_groups)} finalized {self.keyword} chain groups.")
    #     return final_validated_groups
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any

from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

class SegmentIdentificationTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.mode = mode

        self.json_format = """
{
    "segmented_chains": {
        "<EXACT_INPUT_CHAIN_KEY>": {
            "justification": "Detailed anatomical or functional reasoning for this partition pattern.",
            "segments": [
                ["link_1", "link_2"],
                ["link_3"],
                ["link_4", "link_5"]
            ]
        }
    }
}"""

        self.structured_format = {
            "type": "object",
            "properties": {
                "segmented_chains": {
                    "type": "object",
                    "description": "A dictionary mapping each chain ID to its justification and ordered list of segments.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "justification": {
                                "type": "string",
                                "description": "Anatomical or functional rationale for why links were grouped this way."
                            },
                            "segments": {
                                "type": "array",
                                "description": "An ordered list of segments (contiguous link arrays).",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "required": ["justification", "segments"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["segmented_chains"],
            "additionalProperties": False
        }

    def ollama_json_schema(self) -> dict:
        return self.structured_format

    def system_prompt(self) -> str:
        task_prompt = """
You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task:
Partition each provided chain of links into functional segments. A segment is a contiguous group of links representing a distinct anatomical or functional part. For instance, severance links can compose a sub-component (segment) of a limb.
Provide a clear 'justification' explaining the functional grouping rationale for each chain.
"""
        instruction_prompt = """ 
Segmentation Rules:
1. Exact Keys: Under "segmented_chains", you MUST use the EXACT keys provided in the input dictionary.
2. Strict Link Scope: Partition ONLY the links listed in 'allowed_links_to_partition'.
3. Contiguity: Links in a segment MUST be contiguous in the exact order provided in 'allowed_links_to_partition'.
4. Complete Coverage: Every link in 'allowed_links_to_partition' MUST appear in exactly one segment.
5. Disjoint Segments: Links cannot overlap across multiple segments.
6. Justification Required: Explain briefly why links were merged.
"""

        input_prompt = self.get_input_prompt(self.mode)
         
        output_prompt = f"""
Output Format:
Respond strictly in JSON following this structure: {self.json_format}
Do not include any extra text or markdown formatting."""

        return "".join([task_prompt, instruction_prompt, input_prompt, output_prompt])

    def user_prompt(self, user_input: Dict[str, Any]) -> str:
        self.initial_chains = {}
        data_input = {}
        
        for key, identified_chain in user_input.items():
            self.initial_chains[key] = identified_chain.link_chain 
            kinematic_chain_payload = identified_chain.get_kinematic_chain(self.mode)
            
            data_input[key] = {
                "type": identified_chain.type, 
                "allowed_links_to_partition": identified_chain.link_chain,
                "kinematic_chain": kinematic_chain_payload
            }
            
        return f"""
    Below are the kinematic chains for segment identification.

    CRITICAL INSTRUCTION: 
    For each chain, partition ONLY the links explicitly listed under 'allowed_links_to_partition'. 
    Include a concise 'justification' field explaining your structural/functional decisions.

    Input data:
    {json.dumps(data_input, indent=2)}"""

    def interpret(self, output: dict) -> List[Dict[str, Any]]:
        valid_results: List[Dict[str, Any]] = []
        segmented_chains = output.get("segmented_chains", {})

        if not isinstance(segmented_chains, dict):
            return valid_results

        initial_keys = list(self.initial_chains.keys())

        for idx, (chain_id, content) in enumerate(segmented_chains.items()):
            ref_chain = self.initial_chains.get(chain_id)
            target_key = chain_id
            
            if ref_chain is None and idx < len(initial_keys):
                target_key = initial_keys[idx]
                ref_chain = self.initial_chains[target_key]

            if not ref_chain or not isinstance(content, dict):
                continue

            segments = content.get("segments", [])
            justification = content.get("justification", "No justification provided.")

            if not isinstance(segments, list):
                continue

            valid_segments, used_links, is_valid = [], set(), True

            for segment in segments:
                if not isinstance(segment, list) or not segment:
                    is_valid = False
                    break

                if not all(l in ref_chain for l in segment):
                    is_valid = False
                    break

                sorted_links = sorted(segment, key=ref_chain.index)
                indices = [ref_chain.index(l) for l in sorted_links]
                
                is_contiguous = all(idx2 - idx1 == 1 for idx1, idx2 in zip(indices, indices[1:]))
                has_overlap = bool(used_links.intersection(sorted_links))

                if not is_contiguous or has_overlap:
                    is_valid = False
                    break

                used_links.update(sorted_links)
                valid_segments.append({"links": sorted_links})

            if is_valid and used_links == set(ref_chain):
                valid_results.append({"chain_id": target_key, "chain": ref_chain, "justification": justification,"identified_segments": valid_segments})

        print(f"Interpreted {len(valid_results)} validated anonymous partitions.")
        return valid_results

    def aggregate(self, generations: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregates voting patterns while mapping the winning justification."""
        chain_pattern_votes = defaultdict(lambda: defaultdict(int))
        pattern_justifications = {}

        for gen in generations:
            for c in gen:
                chain_id = c["chain_id"]
                pattern = tuple(tuple(seg["links"]) for seg in c["identified_segments"])
                
                chain_pattern_votes[chain_id][pattern] += 1
                # Store the most recent non-empty justification for this pattern
                if c.get("justification"):
                    pattern_justifications[(chain_id, pattern)] = c["justification"]

        final_chains = []
        for chain_id, pattern_counts in chain_pattern_votes.items():
            if not pattern_counts:
                continue

            # Pick the winning pattern with majority vote
            winner_pattern = max(pattern_counts, key=pattern_counts.get)
            winning_justification = pattern_justifications.get(
                (chain_id, winner_pattern), 
                "Consensus segment partition based on majority voting."
            )

            final_chains.append({
                "chain_id": chain_id,
                "chain": self.initial_chains[chain_id],
                "justification": winning_justification,
                "identified_segments": [{"links": list(seg)} for seg in winner_pattern]
            })

        print(f"Aggregated segment selections across {len(final_chains)} chains.")
        return final_chains
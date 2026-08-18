from tasks.VotingTask import VotingTask
from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import json

from tasks.URDFLinkTask import URDFLinkTask

class SegmentIdentificationTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.mode = mode

        self.json_format = """
{
    "segmented_chains": {
        "chain_id": [
            ["link_1", "link_2"],
            ["link_3"],
            ["link_4", "link_5"]
        ]
    }
}"""

        self.structured_format = {
            "type": "object",
            "properties": {
                "segmented_chains": {
                    "type": "object",
                    "description": "A dictionary mapping each chain ID to an ordered list of anonymous segments.",
                    "additionalProperties": {
                        "type": "array",
                        "description": "An ordered list of segments, where each segment is a contiguous array of link name strings.",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["segmented_chains"],
            "additionalProperties": False
        }

    def system_prompt(self) -> str:

        input_prompt, task_prompt, instruction_prompt, output_prompt = "", "", "", ""
   
        task_prompt = f"""
You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task:
Partition the provided chains of links into functional segments. Each segment can contain several links if they account for a mutual anatomical part.
A segment is a contiguous group of links representing a distinct anatomical part. 
"""
        instruction_prompt = f""" 
Segmentation Rules:
1. Contiguity: Links in a segment MUST be contiguous in the provided sequence. 
2. Complete Coverage: Every link in the provided chain must be assigned to exactly one segment.
3. No Overlaps: Links cannot appear in multiple segments."""
# 2. Boundary Detection: Look for keywords in link/joint names to identify transition points between segments.

        input_prompt = self.get_input_prompt(self.mode)
         
        output_prompt = f"""
Output Format:
Respond strictly in JSON following this structure: {self.json_format}
Do not include any extra text or markdown formatting."""

        complete_prompt = "".join([task_prompt, instruction_prompt, input_prompt, output_prompt])
        
        return complete_prompt
    
    def user_prompt(self, user_input: Dict[str, List[str]]) -> str:
        self.initial_chains = {}
        data_input = {}
        
        for link, identified_chain in user_input.items():
            self.initial_chains[link] = identified_chain.link_chain # Store only the base link_chain for verification passes
            kinematic_chain_payload = identified_chain.get_kinematic_chain(self.mode)
            data_input[link] = {
                "type": identified_chain.type, 
                "kinematic_chain": kinematic_chain_payload
            }
            
        new_message = f"""
Below are the kinematic chains for segment identification with their already classified types:

Input data:
{json.dumps(data_input, indent=2)}"""
        
        return new_message
    
    def interpret(self, output: dict) -> List[Dict[str, Any]]:
        valid_results: List[Dict[str, Any]] = []
        segmented_chains = output.get("segmented_chains", {})

        if not isinstance(segmented_chains, dict):
            return valid_results

        for chain_id, segments in segmented_chains.items():
            ref_chain = self.initial_chains.get(chain_id)
            if not ref_chain or not isinstance(segments, list):
                continue

            valid_segments, used_links, is_valid = [], set(), True

            for links in segments:
                if not isinstance(links, list) or not links or not all(l in ref_chain for l in links):
                    is_valid = False
                    break

                # Sort to fix directionality, then assert a strict forward stride of 1 index
                sorted_links = sorted(links, key=ref_chain.index)
                indices = [ref_chain.index(l) for l in sorted_links]
                
                if any(idx2 - idx1 != 1 for idx1, idx2 in zip(indices, indices[1:])) or used_links.intersection(sorted_links):
                    is_valid = False
                    break

                used_links.update(sorted_links)
                valid_segments.append({"links": sorted_links})

            if is_valid and used_links == set(ref_chain):
                valid_results.append({"chain_id": chain_id, "chain": ref_chain, "identified_segments": valid_segments})

        print(f"Interpreted {len(valid_results)} validated anonymous partitions.")
        return valid_results
    
    def aggregate(self, generations: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Simplified voting engine that directly reconstructs the winning segment pattern."""
        chain_votes = defaultdict(list)

        # Step 1: Collect every model's partition choice for each chain
        for gen in generations:
            for c in gen:
                # Store the segments as a list of sorted lists so it's hashable/comparable
                pattern = tuple(tuple(seg["links"]) for seg in c["identified_segments"])
                chain_votes[c["chain_id"]].append(pattern)

        # Step 2: Pick the most frequent pattern per chain and format the output
        final_chains = []
        for chain_id, patterns in chain_votes.items():
            if not patterns:
                continue

            # Find the winner using standard Python counting
            winner_pattern = max(set(patterns), key=patterns.count)

            # Build the output structure directly from the winning pattern slices
            final_chains.append({
                "chain_id": chain_id,
                "chain": self.initial_chains[chain_id],
                "identified_segments": [{"links": list(seg)} for seg in winner_pattern]
            })

        print(f"Aggregated segment selections across {len(final_chains)} chains.")
        return final_chains
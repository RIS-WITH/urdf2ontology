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
        "<EXACT_INPUT_CHAIN_KEY>": [
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
    
    def ollama_json_schema(self) -> dict:
        """Returns the JSON schema expected by Ollama / structured decoding."""
        return self.structured_format

    def system_prompt(self) -> str:
        task_prompt = """
You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task:
Partition each provided chain of links into functional segments. A segment is a contiguous group of links representing a distinct anatomical or functional part.
"""
        instruction_prompt = """ 
Segmentation Rules:
1. Exact Keys: Under "segmented_chains", you MUST use the EXACT keys provided in the input dictionary.
2. Strict Link Scope: Partition ONLY the links listed in 'allowed_links_to_partition' for that specific key. Do NOT add external/parent links (e.g., 'torso_lift_link') if they are not in 'allowed_links_to_partition'.
3. Contiguity: Links in a segment MUST be contiguous in the exact order provided in 'allowed_links_to_partition'.
4. Complete Coverage: Every link in 'allowed_links_to_partition' MUST appear in exactly one segment.
5. Disjoint Segments: Links cannot overlap across multiple segments.
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
            # Store base link chain for validation passes
            self.initial_chains[key] = identified_chain.link_chain 
            
            # Get kinematic chain payload
            kinematic_chain_payload = identified_chain.get_kinematic_chain(self.mode)
            
            data_input[key] = {
                "type": identified_chain.type, 
                "allowed_links_to_partition": identified_chain.link_chain,
                "kinematic_chain": kinematic_chain_payload
            }
            
        return f"""
    Below are the kinematic chains for segment identification.

    CRITICAL INSTRUCTION: 
    For each chain, you MUST ONLY partition the links explicitly listed under 'allowed_links_to_partition'. 
    Do NOT include parent links like 'torso_lift_link' or base links unless they are explicitly present in 'allowed_links_to_partition'.

    Input data:
    {json.dumps(data_input, indent=2)}"""
    
    def interpret(self, output: dict) -> List[Dict[str, Any]]:
        valid_results: List[Dict[str, Any]] = []
        segmented_chains = output.get("segmented_chains", {})

        if not isinstance(segmented_chains, dict):
            return valid_results

        initial_keys = list(self.initial_chains.keys())

        for idx, (chain_id, segments) in enumerate(segmented_chains.items()):
            # 1. Primary Lookup by exact Key; Fallback by positional index
            ref_chain = self.initial_chains.get(chain_id)
            target_key = chain_id
            
            if ref_chain is None and idx < len(initial_keys):
                target_key = initial_keys[idx]
                ref_chain = self.initial_chains[target_key]

            if not ref_chain or not isinstance(segments, list):
                print(f"[Validation Failed] Could not resolve chain key: {chain_id}")
                continue

            valid_segments, used_links, is_valid = [], set(), True

            for segment in segments:
                if not isinstance(segment, list) or not segment:
                    is_valid = False
                    break

                # Rule Check: Foreign links in segment
                if not all(l in ref_chain for l in segment):
                    print(f"[Validation Failed] {target_key}: Segment {segment} contains links not present in reference chain {ref_chain}")
                    is_valid = False
                    break

                # Sort to preserve order relative to reference chain
                sorted_links = sorted(segment, key=ref_chain.index)
                indices = [ref_chain.index(l) for l in sorted_links]
                
                # Rule Check: Contiguity & Overlap
                is_contiguous = all(idx2 - idx1 == 1 for idx1, idx2 in zip(indices, indices[1:]))
                has_overlap = bool(used_links.intersection(sorted_links))

                if not is_contiguous or has_overlap:
                    print(f"[Validation Failed] {target_key}: Segment {segment} violates contiguity or overlaps.")
                    is_valid = False
                    break

                used_links.update(sorted_links)
                valid_segments.append({"links": sorted_links})

            # Rule Check: Complete Coverage
            if is_valid:
                if used_links == set(ref_chain):
                    valid_results.append({
                        "chain_id": target_key, 
                        "chain": ref_chain, 
                        "identified_segments": valid_segments
                    })
                else:
                    missing = set(ref_chain) - used_links
                    print(f"[Validation Failed] {target_key}: Incomplete coverage. Missing links: {missing}")

        print(f"Interpreted {len(valid_results)} validated anonymous partitions.")
        return valid_results
    
    def aggregate(self, generations: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Simplified voting engine that directly reconstructs the winning segment pattern."""
        chain_votes = defaultdict(list)

        for gen in generations:
            for c in gen:
                pattern = tuple(tuple(seg["links"]) for seg in c["identified_segments"])
                chain_votes[c["chain_id"]].append(pattern)

        final_chains = []
        for chain_id, patterns in chain_votes.items():
            if not patterns:
                continue

            winner_pattern = max(set(patterns), key=patterns.count)
            final_chains.append({
                "chain_id": chain_id,
                "chain": self.initial_chains[chain_id],
                "identified_segments": [{"links": list(seg)} for seg in winner_pattern]
            })

        print(f"Aggregated segment selections across {len(final_chains)} chains.")
        return final_chains
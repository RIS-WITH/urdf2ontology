from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from typing import Dict, List, Tuple, Any, Union
import json
from collections import Counter, defaultdict
from AbstractedRobotLinks import TreeComponent, ChainComponent

class LinkNamingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs: int, keyword: str):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "generated_names": {
        "EXACT_COMPONENT_ID": {
            "component_name": "clean_snake_case_macro_name",
            "sub_components": {
                "EXACT_BRANCH_ID": "clean_snake_case_branch_name",
                "EXACT_SEGMENT_ID": "clean_snake_case_segment_name"
            }
        }
    }
}"""

        self.structured_format = {
            "type": "object",
            "properties": {
                "generated_names": {
                    "type": "object",
                    "description": "Mapping of macro entity IDs to their newly generated names and structural sub-components.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "component_name": {
                                "type": "string",
                                "description": "Clean semantic snake_case name for the macro entity."
                            },
                            "sub_components": {
                                "type": "object",
                                "description": "Mapping of individual segment/branch IDs to their clean sub-names.",
                                "additionalProperties": {"type": "string"}
                            }
                        },
                        "required": ["component_name", "sub_components"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["generated_names"],
            "additionalProperties": False
        }
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format
    
class ComponentLinkNamingTask(LinkNamingTask):
    def __init__(self, nb_runs: int, keyword: str, naming_type: str, mode: str = "links"):
        super().__init__(nb_runs, keyword)
        self.naming_type = naming_type  # e.g., "tree_component", "chain_component", or "segment"
        self.mode = mode

    @staticmethod
    def _get_field(obj: Union[dict, Any], field_names: List[str], default: Any = None) -> Any:
        """Helper to extract field values from either a dict or an object instance."""
        if isinstance(obj, dict):
            for name in field_names:
                if name in obj:
                    return obj[name]
            return default
        for name in field_names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _extract_kinematic_chain(self, sub_obj: Any) -> List[str]:
        """Safely extracts a list of link strings regardless of object type (ChainComponent or SegmentElement)."""
        if hasattr(sub_obj, "get_kinematic_chain") and callable(getattr(sub_obj, "get_kinematic_chain")):
            chain = sub_obj.get_kinematic_chain(self.mode)
        else:
            chain = self._get_field(sub_obj, ["link_chain", "all_links", "csv_kinematic_sequence"], [])

        if isinstance(chain, (set, tuple)):
            chain = sorted(list(chain))
        return chain or []

    def system_prompt(self) -> str:
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction. Your task is to generate minimal, human-understandable semantic identifiers for abstracted macro entities and their sub-component segments.

STRICT ID RULES:
1. Exact Key Matching: You MUST retain the EXACT input keys for macro components and ALL sub-component keys (including both branch IDs and nested segment IDs) inside 'sub_components'.
2. Flat Mapping for Sub-Components: In your JSON response under 'sub_components', provide a direct flat key-value pair mapping for every branch ID AND every nested segment ID listed in the input.
3. NO Synthetic Chains: Do NOT invent key names outside the provided input.
4. Complete Coverage: Provide a name for every macro component, branch, and segment key present in the input.

NAMING CONVENTIONS:
1. Namespace Consistency: Use strict snake_case format exclusively (lowercase letters, numbers, underscores).
2. Minimal & Clean: Remove raw URDF boilerplate, location prefixes, or mounting joint labels if functional meaning is preserved (e.g., use 'right_shoulder' instead of 'right_upper_shoulder_mount_link').
3. Unique Names: Every generated macro name and sub-component name must be distinct and semantically meaningful.

Output Format:
Respond strictly in JSON following this exact structure:
{self.json_format}
Do not include any extra text, explanations, or markdown formatting outside the JSON block."""

    def user_prompt(self, user_input: Union[Dict[str, Any], List[Any]]) -> str:
        data_input: Dict[str, Any] = {}
        # Tracking: { elem_id: set(all_valid_sub_ids_and_seg_ids) }
        self.input_verification = defaultdict(set)

        items = user_input.values() if isinstance(user_input, dict) else user_input

        for elem in items:
            elem_id = getattr(elem, "id", "")
            if not elem_id:
                continue

            entry_input = {
                "macro_type": getattr(elem, "type", ""),
                "attachment_root": getattr(elem, "attachment_root", ""),
                "assembly_base_link": getattr(elem, "assembly_base_link", None),
                "sub_components": {}
            }

            is_tree = isinstance(elem, TreeComponent) or hasattr(elem, "branches")
            sub_components = elem.branches if is_tree else getattr(elem, "segments", [])

            for sub in sub_components:
                sub_id = getattr(sub, "id", "")
                if not sub_id:
                    continue

                self.input_verification[elem_id].add(sub_id)
                sub_segments = getattr(sub, "segments", [])
                
                kinematic_chain = (
                    sub.get_kinematic_chain(self.mode) 
                    if hasattr(sub, "get_kinematic_chain") 
                    else list(getattr(sub, "link_chain", []) or getattr(sub, "all_links", []))
                )

                sub_entry = {
                    "semantic_type": getattr(sub, "type", ""),
                    "kinematic_chain": kinematic_chain
                }

                if sub_segments:
                    branch_segments = {}
                    for seg in sub_segments:
                        seg_id = getattr(seg, "id", "")
                        if not seg_id:
                            continue
                        
                        self.input_verification[elem_id].add(seg_id)
                        seg_chain = (
                            seg.get_kinematic_chain(self.mode) 
                            if hasattr(seg, "get_kinematic_chain") 
                            else list(getattr(seg, "link_chain", []))
                        )
                        branch_segments[seg_id] = {
                            "semantic_type": getattr(seg, "type", ""),
                            "kinematic_chain": seg_chain
                        }

                    sub_entry["segments"] = branch_segments

                entry_input["sub_components"][sub_id] = sub_entry

            data_input[elem_id] = entry_input

        components_json = json.dumps(data_input, indent=2)

        return f"""
Below are the {self.keyword} macro components and their individual sub-components to name:

Input data:
{components_json}
"""

    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        generated_names = output.get("generated_names", {})

        if not isinstance(generated_names, dict):
            print("[PARSE ERROR] 'generated_names' missing or is not a valid JSON object.")
            return results

        for comp_id, name_data in generated_names.items():
            if comp_id not in self.input_verification:
                print(f"[REJECTED] Hallucinated/unmatched parent entity ID skipped: '{comp_id}'")
                continue

            comp_name = name_data.get("component_name")
            if comp_name:
                results.append((comp_id, str(comp_name)))

            sub_components = name_data.get("sub_components", {})
            if isinstance(sub_components, dict):
                for sub_id, sub_name in sub_components.items():
                    # Validate sub_id against registered set for parent comp_id
                    if sub_name and sub_id in self.input_verification[comp_id]:
                        results.append((sub_id, str(sub_name)))
                    else:
                        print(f"[REJECTED] Sub-component ID '{sub_id}' dropped or misaligned under parent '{comp_id}'")

        print(f"Interpreted {len(results)} valid {self.keyword} structural names.")
        return results

    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:
        per_chain_votes: Dict[str, List[str]] = defaultdict(list)
        final: Dict[str, str] = {}

        for model_run in votes:
            for target_id, generated_name in model_run:
                per_chain_votes[target_id].append(generated_name)

        for target_id, vote_list in per_chain_votes.items():
            most_common_data = Counter(vote_list).most_common(1)

            if not most_common_data:
                continue

            winner, count = most_common_data[0]
            majority_threshold = (self.nb_runs // 2) + 1

            if count >= majority_threshold:
                final[target_id] = winner
            else:
                print(f"Warning: No strict majority for {target_id}. Top vote was '{winner}' ({count}/{self.nb_runs} runs)")
                final[target_id] = winner

        print(f"Validated {len(final)} {self.keyword} semantic names via majority voting.")
        return final
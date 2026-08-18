from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from typing import Dict, List, Tuple, Any, Union
import json
from collections import Counter, defaultdict

class LinkNamingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs: int, keyword: str):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "generated_names": {
        "EXACT_COMPONENT_ID": {
            "component_name": "clean_snake_case_macro_name",
            "sub_components": {
                "EXACT_SUB_COMP_ID": "clean_snake_case_sub_name"
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
        self.naming_type = naming_type  # e.g., "tree_component" or "chain_component"
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

    def system_prompt(self) -> str:
        return f"""
You are an expert in robot kinematics and URDF-to-Ontology abstraction. Your task is to generate minimal, human-understandable semantic identifiers for abstracted macro entities and their sub-component segments.

STRICT ID RULES:
1. Exact Key Matching: You MUST retain the EXACT input keys for macro components (e.g., 'arm_entity_4', 'torso_entity_0') and sub-components (e.g., 'segment_6', 'segment_7').
2. NO Synthetic Chains: Do NOT invent key names like 'chain_0', 'chain_1', or 'component_0'. Only use the provided input IDs.
3. Complete Coverage: Provide a name for every macro component and every sub-component segment listed in the input.

NAMING CONVENTIONS:
1. Namespace Consistency: Use strict snake_case format exclusively (lowercase letters, numbers, underscores).
2. Minimal & Clean: Remove raw URDF boilerplate, location prefixes, or mounting joint labels if functional meaning is preserved (e.g., use 'right_shoulder' instead of 'right_upper_shoulder_mount_link').
3. Unique Names: Every generated macro name and sub-component name within a component must be distinct and semantically meaningful.

Output Format:
Respond strictly in JSON following this exact structure:
{self.json_format}
Do not include any extra text, explanations, or markdown formatting outside the JSON block."""

    def user_prompt(self, user_input: Union[dict, list]) -> str:
        data_input = {}
        self.input_verification = {}
        
        # Handle dict or list inputs safely
        items = user_input.values() if isinstance(user_input, dict) else user_input

        for elem in items:
            comp_id = self._get_field(elem, ["entity_id", "id", "component_id"])
            if not comp_id:
                continue

            macro_type = self._get_field(elem, ["macro_type", "type", "semantic_type"], "Component")
            mount_link = self._get_field(elem, ["mount_link", "attachment_root", "mounting_link"], "base")
            
            # Select sub-components list (segments or branches)
            if self.naming_type == "tree_component":
                sub_list = self._get_field(elem, ["branches", "segments"], [])
            else:
                sub_list = self._get_field(elem, ["segments", "branches"], [])

            entry_input = {
                "macro_type": macro_type,
                "mounting_link": mount_link,
                "sub_components": {}
            }
            
            self.input_verification[comp_id] = set()

            for sub in sub_list:
                sub_id = self._get_field(sub, ["id", "segment_id", "branch_id"])
                if not sub_id:
                    continue

                sub_type = self._get_field(sub, ["type", "semantic_type"], "Subpart")
                
                # Extract kinematic chain or link sequence if available
                if hasattr(sub, "get_kinematic_chain") and callable(getattr(sub, "get_kinematic_chain")):
                    kin_chain = sub.get_kinematic_chain(self.mode)
                else:
                    kin_chain = self._get_field(sub, ["link_chain", "csv_kinematic_sequence"], [])

                entry_input["sub_components"][sub_id] = {
                    "semantic_type": sub_type,
                    "kinematic_chain": kin_chain
                }
                self.input_verification[comp_id].add(sub_id)

            data_input[comp_id] = entry_input

        components_json = json.dumps(data_input, indent=2)

        return f"""
Below are the {self.keyword} macro components and their individual sub-component segments to name:

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
            # Filter out hallucinated keys like 'chain_0'
            if comp_id not in self.input_verification:
                print(f"[REJECTED] Hallucinated/unmatched parent entity ID skipped: '{comp_id}'")
                continue

            comp_name = name_data.get("component_name")
            if comp_name:
                results.append((comp_id, str(comp_name)))

            sub_components = name_data.get("sub_components", {})
            if isinstance(sub_components, dict):
                for sub_id, sub_name in sub_components.items():
                    if sub_name and sub_id in self.input_verification[comp_id]:
                        results.append((sub_id, str(sub_name)))
                    else:
                        print(f"[REJECTED] Sub-component ID '{sub_id}' dropped or misaligned under parent '{comp_id}'")

        print(f"Interpreted {len(results)} valid {self.keyword} structural names.")
        return results

    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:
        per_chain_votes: Dict[str, List[str]] = defaultdict(list)
        final: Dict[str, str] = {}

        # 1. Collect all names assigned to each specific ID across all runs
        for model_run in votes:
            for target_id, generated_name in model_run:
                per_chain_votes[target_id].append(generated_name)

        # 2. Run the majority voting check per ID
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
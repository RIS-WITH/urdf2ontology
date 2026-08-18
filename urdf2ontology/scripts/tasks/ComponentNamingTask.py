from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from typing import Dict, List, Tuple, Any
import json
from collections import Counter, defaultdict

class LinkNamingTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "generated_names": {
        "COMPONENT_ID": {
            "component_name": "upper_assembly_name",
            "sub_components": {
                "SUB_COMP_ID_1": "sub_component_name_1"
            }
        }
    }
}"""

        self.structured_format = {
            "type": "object",
            "properties": {
                "generated_names": {
                    "type": "object",
                    "description": "Mapping of root component IDs to their newly generated upper names and structural sub-components.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "component_name": {
                                "type": "string",
                                "description": "Clean semantic snake_case name for the upper cluster."
                            },
                            "sub_components": {
                                "type": "object",
                                "description": "Mapping of individual sub-component branch IDs to their clean sub-names.",
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
    # default mode is with 'links' -> link_chain
    def __init__(self, nb_runs: int, keyword: str, naming_type: str, mode: str = "links"):
        super().__init__(nb_runs, keyword)
        self.naming_type = naming_type
        self.mode = mode
                
    def system_prompt(self) -> str:
        type_context = ("continuous multi-segment kinematics chains" if self.naming_type == "chain_component" else "branched tree structures")
       
        return f"""
You are an expert in robot kinematics and URDF-to-Ontology abstraction. Your task is to generate a name for all individuals that will populate an ontology representing an abstracted robot. Those names must be minimal, human-understandable semantic identifiers for abstracted {self.keyword} components and sub-components.

Naming Conventions:
1. Namespace Consistency: Use strict snake_case format exclusively. Lowercase letters, numbers, and underscores only.
2. Keep it Minimal: Strip away standard URDF boilerplate words if the remaining words are sufficient to functionally identify the element.
3. Uniqueness: Each generated component and sub-component name must be unique as they will be used to create different individuals in the ontology.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
Do not include any extra text or markdown formatting."""

    def user_prompt(self, user_input) -> str:
        data_input = {}
        self.input_verification = {}
            
        for elem in user_input.values():
            entry_input = {
                "semantic_type": elem.type,
                "mounting_link": elem.attachment_root,
                "sub_components": {}
            }
            
            self.input_verification[elem.id] = set()
            
            if self.naming_type == "tree_component":
                sub_components = elem.branches
            elif self.naming_type == "chain_component":
                sub_components = elem.segments
            else:
                sub_components = []
                
            for sub in sub_components:
                branch_input = {
                    "semantic_type": sub.type,
                    "kinematic_chain": sub.get_kinematic_chain(self.mode)
                }
                entry_input["sub_components"][sub.id] = branch_input
                self.input_verification[elem.id].add(sub.id)
                
            data_input[elem.id] = entry_input
        
        components = json.dumps(data_input, indent=2)
        
        message = f"""
Below are the {self.keyword} components to name:

Input data: {components}
"""
        return message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        generated_names = output.get("generated_names", {})
        
        if not isinstance(generated_names, dict):
            print("[PARSE ERROR] 'generated_names' missing or is not a valid JSON object.")
            return results
        
        for comp_id, name_data in generated_names.items():
            if comp_id not in self.input_verification:
                print(f"[UNKNOWN TRACK] Hallucinated parent component ID skipped: '{comp_id}'")
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
                        print(f"[REJECTED TRACK] Sub-component ID '{sub_id}' was dropped or misaligned under root: '{comp_id}'")
                    
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
                print(f"Warning: No majority for {target_id}. Top vote was '{winner}' ({count}/{self.nb_runs} runs)")
                # Fallback to the top vote even if it didn't cross the strict majority threshold
                final[target_id] = winner

        print(f"Validated {len(final)} {self.keyword} semantic names via majority voting.")
        return final

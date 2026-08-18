from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict
import json


class LinkClassificationTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
                
        self.json_format = """
{
    "classifications": {
        "target_id": {
            "type": "TypeFromAllowedList", 
            "justification": "Clear single-sentence justification"
        }
    }
}"""

        self.structured_format = {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "object",
                    "description": f"A dictionary mapping each target {self.keyword} ID to its classification data.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "The classified ontological subclass matching the allowed types list."
                            },
                            "justification": {
                                "type": "string",
                                "description": "A single self-sufficient justification sentence."
                            }
                        },
                        "required": ["type", "justification"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["classifications"],
            "additionalProperties": False
        }
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format


class ComponentLinkSingleClassificationTask(LinkClassificationTask):

    def __init__(self, nb_runs: int, keyword: str, mode: str):
        super().__init__(nb_runs, keyword)
        self.valid_types = [self.camel_case_keyword(keyword)]
        self.parent_context = self.camel_case_keyword(keyword)
        self.mode = mode
        self.initial_chains: Dict[str, Any] = {}
        self.key_aliases: Dict[str, str] = {}

    def camel_case_keyword(self, keyword) -> str:
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        return "".join(new_words)
             
    def system_prompt(self) -> str:
        task_prompt = f""" 
You are an expert in robot kinematics and URDF-to-Ontology abstraction.
        
Task: You will be provided with target {self.keyword} link chains along with their attachment roots and kinematic details. 
Assign a functional sub-type to each {self.keyword} based on its role.
Context: All target elements are confirmed children of the category: '{self.parent_context}'. 

Allowed Types:
{self.valid_types}
"""
        
        instruction_prompt = f""" 
Classification Rules:
1. Role-Based Classification: Select the type that accurately reflects the kinematic chain's specific function.
2. Structural Analysis: Use the relationship between the links and the attachment root to determine functional purpose.
3. Constraint: Choose exactly ONE type from the allowed list.
4. Convergence Fallback: Use the parent category '{self.parent_context}' if no sub-type is sufficiently supported by the evidence.
5. Key Continuity: You MUST use the exact target ID key provided in the input JSON for each entry in 'classifications'.
"""
        
        input_prompt = self.get_input_prompt(self.mode)
         
        output_prompt = f"""
Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting.
"""
        return "".join([task_prompt, instruction_prompt, input_prompt, output_prompt])
        
    def user_prompt(self, user_input: dict) -> str:
        self.initial_chains = {}
        self.key_aliases = {}
        data_input = {}
        
        for target_key, identified_comp in user_input.items():
            # Support both object attributes and dictionary keys
            link_chain = getattr(identified_comp, "link_chain", None) or identified_comp.get("link_chain", [])
            attachment_root = getattr(identified_comp, "attachment_root", None) or identified_comp.get("attachment_root")
            
            # Store primary target chain
            self.initial_chains[target_key] = link_chain
            
            # Alias mapping: map primary ID, link frames, and end links back to the target_key
            self.key_aliases[target_key] = target_key
            for link in link_chain:
                self.key_aliases[link] = target_key
                
            end_link = getattr(identified_comp, "segment_end_link", None) or getattr(identified_comp, "assembly_base_link", None)
            if end_link:
                self.key_aliases[end_link] = target_key

            # Get kinematic chain payload
            if hasattr(identified_comp, "get_kinematic_chain"):
                kin_chain = identified_comp.get_kinematic_chain(self.mode)
            else:
                kin_chain = identified_comp.get("kinematic_chain", link_chain)

            data_input[target_key] = {
                "attachment_root": attachment_root, 
                "kinematic_chain": kin_chain
            }
            
        chains = json.dumps(data_input, indent=2)
        
        return f"""
Below are the identified target {self.keyword} components with their associated kinematic subchain context:

Input data:
{chains}"""

    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        
        if not isinstance(output, dict):
            return results
            
        items_dict = output.get("classifications", {})
        if not isinstance(items_dict, dict):
            print("[PARSE ERROR] 'classifications' key missing or is not a valid JSON dictionary.")
            return results

        processed_canonical_keys = set()

        for raw_item_id, entry_dict in items_dict.items():
            if not isinstance(entry_dict, dict):
                continue
                
            assigned_type = entry_dict.get("type")
            
            if not raw_item_id or not assigned_type:
                continue

            # Resolve key via alias table if LLM used URDF link name instead of segment/branch ID
            canonical_id = self.key_aliases.get(raw_item_id)

            if canonical_id and canonical_id in self.initial_chains:
                if canonical_id in processed_canonical_keys:
                    continue  # Prevent duplicate mappings if LLM lists both link and segment ID
                    
                if assigned_type in self.valid_types:
                    results.append((canonical_id, assigned_type))
                    processed_canonical_keys.add(canonical_id)
                else:
                    print(f"[INVALID CATEGORY DROPPED] Target '{raw_item_id}' (resolved as '{canonical_id}') assigned out-of-bounds type: '{assigned_type}'")
            else:
                print(f"[UNKNOWN TARGET DROPPED] Model returned classification for untracked ID: '{raw_item_id}'")
                
        print(f"Interpreted {len(results)} verified {self.keyword} classifications.")
        return results

    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:
        per_link_votes = defaultdict(list)

        for model_run_votes in votes:
            for link, assigned_type in model_run_votes:
                per_link_votes[link].append(assigned_type)
                
        final_consensus: Dict[str, str] = {}
        majority_threshold = (len(votes) // 2) + 1 if len(votes) > 1 else 1

        for link, types in per_link_votes.items():
            most_common_data = Counter(types).most_common(1)
            
            if not most_common_data:
                final_consensus[link] = f"Unknown{self.parent_context}"
                continue
                
            most_common, count = most_common_data[0]
            
            if count >= majority_threshold:
                final_consensus[link] = most_common
            else:
                fallback_type = f"Unknown{self.parent_context}"
                print(f"[CONSENSUS SPLIT] Target '{link}' had unstable votes: {dict(Counter(types))}. Falling back to '{fallback_type}'")
                final_consensus[link] = fallback_type
        
        print(f"Validated {len(final_consensus)} finalized {self.keyword} classifications via voting.")
        return final_consensus
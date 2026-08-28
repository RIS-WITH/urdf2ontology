from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict
import json
from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask


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
        # Maintain lookup maps across batches
        self.initial_chains: Dict[str, Any] = {}
        self.key_aliases: Dict[str, str] = {}
        self.current_batch_input_keys: List[str] = []

    def camel_case_keyword(self, keyword) -> str:
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        return "".join(new_words)
             
    def system_prompt(self) -> str:
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.

Task: You will be provided with target {self.keyword} link chains along with their attachment roots, link names, and kinematic details. 
Assign a functional sub-type to each {self.keyword} based ONLY on explicit, unambiguous evidence in the provided data.

Context: All target elements are confirmed instances/children of the parent category: '{self.parent_context}'. 

Allowed Types:
{self.valid_types}

Classification Rules:
1. Strict Grounding (No Over-Interpretation): Classify an entity into a specific sub-type ONLY if explicit naming or structural evidence strictly confirms it.
   - For Sensors: Generic names do NOT necessarily provide enough evidence to infer sub-types. Use the broader concept or the parent fallback unless explicit modalities are in the data.
   - For End-Effectors: Do not infer specific actuation types unless explicitly stated.
   - For Locomotion: Do not infer specific propulsion mechanics unless explicit keywords or joint types confirm it.
   - For Body Parts: Do not infer specialized skeletal roles unless explicitly named.
2. Evidence-Based Selection: Select the type that accurately reflects the kinematic chain's functional purpose as supported directly by link names, joint types, or attachment contexts.
3. Strict Fallback Rule: If the input data is ambiguous, generic, or lacks explicit subtype evidence, you MUST fall back to '{self.parent_context}' (or the most generic allowed matching parent concept). NEVER invent, assume, or extrapolate details not present in the input.
4. Constraint: Choose exactly ONE type from the allowed list.
5. Key Continuity: You MUST use the exact target ID key provided in the input JSON for each entry in 'classifications'.

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting."""
        
    def user_prompt(self, user_input: dict) -> str:
        # Clear previous batch mappings to prevent cross-batch contamination
        self.initial_chains = {}
        self.key_aliases = {}
        self.current_batch_input_keys = list(user_input.keys())
        data_input = {}
        
        for target_key, identified_comp in user_input.items():
            link_chain = getattr(identified_comp, "link_chain", None) or identified_comp.get("link_chain", [])
            
            if isinstance(identified_comp, dict):
                attachment_root = identified_comp.get("attachment_root")
            else:
                attachment_root = getattr(identified_comp, "attachment_root", None)
                
            self.initial_chains[target_key] = link_chain
            
            # Map primary ID and sub-links back to canonical target_key
            self.key_aliases[target_key] = target_key
            for link in link_chain:
                self.key_aliases[link] = target_key
                
            end_link = getattr(identified_comp, "segment_end_link", None) or getattr(identified_comp, "assembly_base_link", None)
            if end_link:
                self.key_aliases[end_link] = target_key

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

            canonical_id = self.key_aliases.get(raw_item_id)

            if canonical_id and canonical_id in self.initial_chains:
                if canonical_id in processed_canonical_keys:
                    continue  
                    
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

        # Ensure all items present in the input batch are aggregated (even if missing from responses)
        all_target_keys = set(self.current_batch_input_keys) | set(per_link_votes.keys())

        for link in all_target_keys:
            types = per_link_votes.get(link, [])
            most_common_data = Counter(types).most_common(1)
            
            if not most_common_data:
                final_consensus[link] = self.parent_context
                continue
                
            most_common, count = most_common_data[0]
            
            if count >= majority_threshold:
                final_consensus[link] = most_common
            else:
                fallback_type = self.parent_context
                print(f"[CONSENSUS SPLIT] Target '{link}' had unstable votes: {dict(Counter(types))}. Falling back to '{fallback_type}'")
                final_consensus[link] = fallback_type
        
        print(f"Validated {len(final_consensus)} finalized {self.keyword} classifications via voting.")
        return final_consensus
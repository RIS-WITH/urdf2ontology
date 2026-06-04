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
    "chains": {
        "example_target_link_1": "TypeFromAllowedList",
        "example_target_link_2": "TypeFromAllowedList"
    }
}"""
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "chains": {
                    "type": "object",
                    "description": f"A dictionary mapping each target {self.keyword} ID to its classified type string.",
                    "additionalProperties": {
                        "type": "string",
                        "description": "The classified ontological subclass string matching the allowed types list."
                    }
                }
            },
            "required": ["chains"],
            "additionalProperties": False
        }
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format
 
class ComponentLinkSingleClassificationTask(LinkClassificationTask):

    def __init__(self, nb_runs : int, keyword : str, mode: str):
        super().__init__(nb_runs, keyword)
        self.valid_types = [self.camel_case_keyword(keyword)]
        self.parent_context = self.camel_case_keyword(keyword)
        self.mode = mode

    def camel_case_keyword(self, keyword) -> str:
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        camel_case_keyword = "".join(new_words)
        return camel_case_keyword
             
    def system_prompt(self) -> str:
        task_prompt, input_prompt, instruction_prompt, output_prompt = "", "", "", ""
    
        task_prompt = f""" 
You are an expert in robot kinematics and URDF-to-Ontology abstraction.
        
Task: You will be provided with a list of target {self.keyword} chain of links with their attachment root and their kinematic chain. 
Assign a functional sub-type to each {self.keyword} chain of links based on its role.
Context: All target elements are confirmed children of the category: '{self.parent_context}'. 

Allowed Types:
{self.valid_types}
"""
        
        instruction_prompt = f""" 
Classification Rules:
1. Role-Based Classification: Select the type that accurately reflects the kinematic chain's specific function.
2. Structural Analysis: Use the relationship between the links and the attachment root to determine functional purpose.
3. Constraint: Choose exactly ONE type from the allowed list. Use the parent category '{self.parent_context}' if no sub-type is sufficiently supported by the evidence.
"""
        
        input_prompt = self.get_input_prompt(self.mode)
         
        output_prompt = f"""
Output Format:
Respond strictly in JSON following this structure: {self.json_format}
Do not include any extra text or markdown formatting."""
        
        complete_prompt = "".join([task_prompt, instruction_prompt, input_prompt, output_prompt])
    
        return complete_prompt
        
    def user_prompt(self, user_input: dict) -> str:

        self.initial_chains = {}
        data_input = {}
        
        for link, identified_comp in user_input.items():
            self.initial_chains[link] = identified_comp.link_chain # Store only the base link_chain for verification passes
            
            data_input[link] = {
                "attachment_root": identified_comp.attachment_root, 
                "kinematic_chain": identified_comp.get_kinematic_chain(self.mode)
            }
            
        chains = json.dumps(data_input, indent=2)
        
        new_message = f"""
Below are the identified target {self.keyword} components with their associated kinematic subchain context:

Input data:{chains}"""
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        
        if not isinstance(output, dict):
            return results
            
        items_dict = output.get("chains", {})
        if not isinstance(items_dict, dict):
            print(f"[PARSE ERROR] 'chains' key missing or is not a valid JSON dictionary.")
            return results

        for item_id, assigned_type in items_dict.items():
            if item_id in self.initial_chains:
                if assigned_type in self.valid_types:
                    results.append((item_id, assigned_type))
                else:
                    print(f"[INVALID CATEGORY DROPPED] Target '{item_id}' assigned out-of-bounds type: '{assigned_type}'")
            else:
                print(f"[UNKNOWN TARGET DROPPED] Model returned classification for untracked chain ID: '{item_id}'")
                
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
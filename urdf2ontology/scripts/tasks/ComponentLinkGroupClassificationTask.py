from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

import json
from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict


class LinkClassificationTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
         
        self.json_format ="""
{
    "groups": {
        "group_id_1": "TypeFromAllowedList",
        "group_id_2": "TypeFromAllowedList"
    }
}"""
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "object",
                    "description": f"A dictionary mapping each evaluated {self.keyword} group ID directly to its composite category.",
                    "additionalProperties": {
                        "type": "string",
                        "description": "The classified ontological subclass string matching the allowed types list."
                    }
                }
            },
            "required": ["groups"],
            "additionalProperties": False
        }
        
    def ollama_json_schema(self) -> Dict[str, Any]:
        return self.structured_format
 
class ComponentLinkGroupClassificationTask(LinkClassificationTask):

    def __init__(self, nb_runs : int, keyword : str, mode: str):
        super().__init__(nb_runs, keyword)
        self.valid_types = [self.camel_case_keyword(keyword)]
        self.parent_context = self.camel_case_keyword(keyword)
        self.mode = mode
    
    def camel_case_keyword(self, keyword):
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        camel_case_keyword = "".join(new_words)
        return camel_case_keyword
             
    def system_prompt(self) -> str:
        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.
        
Task: Determine the high-level functional identity of the grouped {self.keyword} component.
Context: These groups are semantic refinements of the core parent category: '{self.parent_context}'.

Allowed Types:
{self.valid_types}

Input Structure:
The component is represented as an aggregated group containing one or several sub-link branches. For each branch link, you are provided with:
1. Semantic Type: The previously classified operational label of the leaf node.
2. Attachment Root: The physical interface link where the entire branch mounts to the robot's main core body.
3. Kinematic Chain: The ordered physical path flowing sequentially from the attachment root to that specific leaf.

Synthesis Logic:
1. Holistic Identity: Infer the final component group type by aggregating and evaluating the collective roles of all internal sub-links.
2. Ontological Alignment: The selected type must be a logically valid specialization of '{self.parent_context}'.
3. Convergence Fallback: If the internal branches do not provide clear, sufficient evidence for a specialized sub-type, retain the generic baseline parent category: '{self.parent_context}'.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in valid JSON matching this schema layout exactly. Do not include markdown code block tags or conversational text:
{self.json_format}"""

    def user_prompt(self, user_input) -> str:
        self.initial_groups = dict.fromkeys(user_input.keys(), "")
        data_input = {}
        
        for group_id, identified_group in user_input.items():
            group_input = []
            for branch in identified_group.branches:
                group_input.append({
                    "semantic_type": getattr(branch, "type", "Unknown"), 
                    "attachment_root": getattr(branch, "attachment_root", "Unknown"), 
                    "kinematic_chain": branch.get_kinematic_chain(self.mode)
                })
            data_input[group_id] = group_input
        
        groups = json.dumps(data_input, indent=2)
        
        new_message = f"""
Below are the identified target {self.keyword} group components and their internal branch data configurations:

Input data: {groups}"""

        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        
        if not isinstance(output, dict):
            return results
            
        items_dict = output.get("groups", {})
        if not isinstance(items_dict, dict):
            print(f"[PARSE ERROR] Required 'groups' dictionary key missing from output.")
            return results

        for item_id, assigned_type in items_dict.items():
            if item_id in self.initial_groups:
                if assigned_type in self.valid_types:
                    results.append((item_id, assigned_type))
                else:
                    print(f"[INVALID TYPE] Group '{item_id}' assigned out-of-bounds class: '{assigned_type}'")
            else:
                print(f"[UNKNOWN TRACK] Model returned a classification for an untracked group configuration: '{item_id}'")
                
        print(f"Interpreted {len(results)} verified {self.keyword} group classifications.")
        return results

    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:
        per_link_votes = defaultdict(list)
        
        for model_votes in votes:
            for link, assigned_type in model_votes:
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
                print(f"[CONSENSUS SPLIT] Group '{link}' had split votes: {dict(Counter(types))}. Defaulting to '{fallback_type}'")
                final_consensus[link] = fallback_type
        
        print(f"Validated {len(final_consensus)} finalized {self.keyword} group classifications via voting.")
        return final_consensus
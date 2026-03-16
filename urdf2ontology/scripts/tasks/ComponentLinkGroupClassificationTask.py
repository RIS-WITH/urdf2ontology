from tasks.VotingTask import VotingTask

import json
from typing import Tuple, List, Dict, Any
from collections import Counter, defaultdict


class LinkClassificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
         
        self.json_format = f"""
        {{
            "{self.keyword}s":
            [
                {{
                    "{self.keyword}_group": "string",
                    "{self.keyword}_type": "one of the predefined types"
                }}
            ]
        }}
        """
        
        self.structured_format = {
        "type": "object",
        "properties": {
            f"{self.keyword}s": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                f"{self.keyword}"+"_group": {"type": "string"},
                f"{self.keyword}"+"_type": {"type": "string"}
                },
                "required": [f"{self.keyword}"+"_group", f"{self.keyword}"+"_type"],
                "additionalProperties": False
            }
            }
        },
        "required": [f"{self.keyword}s"],
        "additionalProperties": False
        }
 
class ComponentLinkGroupClassificationTask(LinkClassificationTask):

    def __init__(self, nb_runs : int, keyword : str):
        super().__init__(nb_runs, keyword)
        self.valid_types = [self.camel_case_keyword(keyword)]
        self.parent_context = self.camel_case_keyword(keyword)
    
    def camel_case_keyword(self, keyword):
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        camel_case_keyword = "".join(new_words)
        return camel_case_keyword
             
    def system_prompt(self) -> str:
    
        compacted_prompt_group = f"""
Task: Determine the high-level functional identity of the grouped {self.keyword} component.
Context: These groups are refinements of the category: '{self.parent_context}'.

Allowed Types:
{self.valid_types}

Synthesis Logic:
1. Holistic Identity: Infer the component type by aggregating the functional roles of all internal sub-links.
2. Ontological Alignment: The selected type must be a logically valid specialization of '{self.parent_context}'.
3. Convergence: If the internal parts do not provide sufficient evidence for a more specific sub-type, retain the current category '{self.parent_context}'.

Output Format: Respond strictly in JSON: {self.json_format}
"""

        return compacted_prompt_group
        
    def user_prompt(self, user_input) -> str:
       
        self.initial_groups = user_input
    
        new_message = f"""
Below are grouped {self.keyword} components. Each group represents a single physical device potentially
composed of multiple frames and links.

Each entry corresponds to one grouped physical device and contains a list of element like below:
- Link: The name of the specific URDF frame.
- Semantic Type: The pre-classified semantic role of that link.
- Mounting Point: The link where this entire device is attached to the robot.
- Subchain: The kinematic structure associated with that specific link.

Your task:
Assign exactly one predefined type to each group.

Groups: {user_input}"""
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        results = []

        items = output.get(f"{self.keyword}s", [])
        if not isinstance(items, list):
            return results

        for item in items:
            group_key = item.get(f"{self.keyword}_group")
            type_key = item.get(f"{self.keyword}_type")
            
            if(group_key in self.initial_groups.keys() and type_key in self.valid_types):
                results.append((group_key, type_key))
                
        print("Output: --interpreted", len(results), f"{self.keyword} groups")
        return results
    
    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:

        per_link_votes = defaultdict(list)
        
        for model_votes in votes:
            for link, sensor_type in model_votes:
                per_link_votes[link].append(sensor_type)
                
        final = {}
        for link, types in per_link_votes.items():
            most_common, count = Counter(types).most_common(1)[0]
            if count >= int(self.nb_runs/2) + 1:
                final[link] = most_common
            else:
                final[link] = "Unknown" + self.camel_case_keyword(self.keyword)
        
        print("Validated ", len(final), f"{self.keyword} groups")
        return final
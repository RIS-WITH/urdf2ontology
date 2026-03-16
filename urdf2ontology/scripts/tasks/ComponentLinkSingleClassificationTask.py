from tasks.VotingTask import VotingTask

from typing import Tuple, List, Dict
from collections import Counter, defaultdict


class LinkClassificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = f"""
        {{
            "{self.keyword}s":
            [
                {{
                    "{self.keyword}_link": "string",
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
                f"{self.keyword}"+"_link": {"type": "string"},
                f"{self.keyword}"+"_type": {"type": "string"}
                },
                "required": [f"{self.keyword}"+"_link", f"{self.keyword}"+"_type"],
                "additionalProperties": False
            }
            }
        },
        "required": [f"{self.keyword}s"],
        "additionalProperties": False
        }
 
class ComponentLinkSingleClassificationTask(LinkClassificationTask):

    def __init__(self, nb_runs : int, keyword : str):
        super().__init__(nb_runs, keyword)
        # self.valid_types = sensor_types
        # self.ontology = sensor_types
        # self.set_camel_case_keyword(keyword)
        self.valid_types = [self.camel_case_keyword(keyword)]
        self.parent_context = self.camel_case_keyword(keyword)

    def camel_case_keyword(self, keyword) -> str:
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        camel_case_keyword = "".join(new_words)
        return camel_case_keyword
             
    def system_prompt(self) -> str:
    
        compacted_prompt_single = f"""
Task: Assign a functional sub-type to each {self.keyword} sub-link based on its role.
Context: All elements are confirmed children of the category: '{self.parent_context}'.

Allowed Types:
{self.valid_types}

Rules:
1. Role-Based Classification: Select the type that accurately reflects the link's specific function within the kinematic sub-chain.
2. Structural Analysis: Use the relationship between the leaf link and the Attachment Root to determine functional purpose.
3. Constraint: Choose exactly ONE type from the allowed list. Use the parent category '{self.parent_context}' if no sub-type is sufficiently supported by the evidence.

Output Format: Respond strictly in JSON: {self.json_format}
"""
            
        return compacted_prompt_single
        
    def user_prompt(self, user_input) -> str:
       
        self.initial_groups = user_input
    
        new_message = f"""
        Below are the {self.keyword} components.

        Each entry corresponds to one physical device and contains:
        - leaf_link
        - intrinsic subchain

        Components: {user_input}"""
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results = []

        items = output.get(f"{self.keyword}s", [])
        if not isinstance(items, list):
            return results

        for item in items:
            link_key = item.get(f"{self.keyword}_link")
            type_key = item.get(f"{self.keyword}_type")
            
            if(link_key in self.initial_groups.keys() and type_key in self.valid_types):
                results.append((link_key, type_key))
                
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
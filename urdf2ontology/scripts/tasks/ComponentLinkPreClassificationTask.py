from tasks.VotingTask import VotingTask

import json
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
                    "{self.keyword}_component": "string",
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
                f"{self.keyword}"+"_component": {"type": "string"},
                f"{self.keyword}"+"_type": {"type": "string"}
                },
                "required": [f"{self.keyword}"+"_component", f"{self.keyword}"+"_type"],
                "additionalProperties": False
            }
            }
        },
        "required": [f"{self.keyword}s"],
        "additionalProperties": False
        }
 
class ComponentLinkPreClassificationTask(LinkClassificationTask):

    def __init__(self, sensor_types: dict, nb_runs : int, keyword : str):
        super().__init__(nb_runs, keyword)
        self.valid_types = self.flatten_ontology(sensor_types)
        self.ontology = sensor_types
        self.set_camel_case_keyword(keyword)
    
    def set_camel_case_keyword(self, keyword):
        words = keyword.split("_")
        new_words = [word.capitalize() for word in words]
        self.camel_case_keyword = "".join(new_words)
    
    def flatten_ontology(self, ontology_dict):
        leaves = set()
        for key, children in ontology_dict.items():
            if not children:
                leaves.add(key)
            else:
                leaves.update(self.flatten_ontology(children))
        return leaves
                         
    def system_prompt(self) -> str:
        compacted_prompt = f"""
        You are an expert in robot modeling and ontology abstraction from URDF descriptions.

        Task:
        Assign exactly ONE type to each identified {self.keyword} component.

        Allowed types:
        {self.valid_types}

        Rules:
        - Choose strictly from allowed types.
        - Do not invent new types.
        - Base decision on leaf names, parent links, and structure.
        - If insufficient evidence and no previous classification, assign Unknown{self.camel_case_keyword}. 
        - Do not leave any identified component unclassified.
        - Apply criteria consistently.

        Example:
        Input: "generic_{self.keyword}_link": {{"previous_classif": "{self.camel_case_keyword}", "chain": ["generic_root_link", "generic_mount_link", "generic_{self.keyword}_link"]}}
        Output: "{self.keyword}s": [{{"{self.keyword}_component": "generic_{self.keyword}_link", "{self.keyword}_type": "Refined{self.camel_case_keyword}"}}]
        
        Output Format:
        Respond only in JSON:
        {self.json_format}
        Do not include explanations or additional text. 
        """
        
        return compacted_prompt
        
    def user_prompt(self, user_input) -> str:
        self.initial_components = user_input

        new_message = f"""
        Below are the identified {self.keyword} components.

        Each entry corresponds to one component and contains:
        - leaf link name
        - intrinsic subchain up to its attachment root
        - previous classification

        Components: {user_input}""" 
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        results = []

        items = output.get(f"{self.keyword}s", [])
        if not isinstance(items, list):
            return results

        for item in items:
            group_key = item.get(f"{self.keyword}_component")
            type_key = item.get(f"{self.keyword}_type")
           
            if(group_key in self.initial_components.keys() and type_key in self.valid_types):
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
                final[link] = "Unknown" + self.camel_case_keyword
        
        print("Validated ", len(final), f"{self.keyword} groups")
        return final


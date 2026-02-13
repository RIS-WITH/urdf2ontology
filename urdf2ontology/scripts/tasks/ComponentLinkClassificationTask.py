from tasks.VotingTask import VotingTask

import json
from typing import Tuple, List, Dict
from collections import Counter, defaultdict


class LinkClassificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
        {
            """ + f"{self.keyword}s" + """: 
            [
                {""" + f"{self.keyword}" + """_group": "string", """ + f"{self.keyword}" + """_type": "one of the predefined types"},
            ]
        }
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
 
class ComponentLinkClassificationTask(LinkClassificationTask):

    def __init__(self, sensor_types: dict, nb_runs : int, keyword : str):
        super().__init__(nb_runs, keyword)
        self.valid_types = self.flatten_ontology(sensor_types) # should be exatcly one because it is the upper category (e.g., Sensor)
        self.ontology = sensor_types
        
    def flatten_ontology(self, ontology_dict):
        leaves = set()
        for key, children in ontology_dict.items():
            if not children:
                leaves.add(key)
            else:
                leaves.update(self.flatten_ontology(children))
        return leaves
                         
    def system_prompt(self) -> str:
        
        new_prompt =  f"""
        You are given grouped {self.keyword} components extracted from a URDF model. Each group corresponds to one physical {self.keyword} device.

        Your task:
        Assign exactly ONE type to each {self.keyword} group.

        Allowed types in the ontology: {self.valid_types} 

        Classification rules:
        - Each group must receive exactly one type.
        - The assigned type must be one of the allowed types.
        - Base your decision on link names, subchain structure, and functional hints.
        - Do NOT invent new types.
        - If uncertain, classify it as UnknownSensor
        - Do not leave any group unclassified.

        Respond only in the following JSON format:
        {self.json_format}
        """
        
        return new_prompt
        
    def user_prompt(self, user_input) -> str:
        # links = json.dumps(user_input["links"], indent=2)
        # message = "Here is the list of links to classify:\n" + links
        
        self.initial_groups = user_input
        groups = json.dumps(user_input, indent=2)
        
        old_message = "Here is the list of sensor groups to classify:\n" + groups
        
        new_message = f"""
        Below are grouped {self.keyword} components.

        Each entry corresponds to one physical device and contains:
        - grouped_leaf_links
        - common_parent
        - intrinsic subchains (if available)

        Your task:
        Assign exactly one predefined type to each group.

        Groups: {groups}"""
        
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        # print("======= Interpreting =======")
        results = []

        items = output.get(f"{self.keyword}s", [])
        if not isinstance(items, list):
            return results

        for item in items:
            group_key = item.get(f"{self.keyword}_group")
            type_key = item.get(f"{self.keyword}_type")
            # check that the link is a str and that the classified sensor type exists in the valid ontology classes
            if(group_key in self.initial_groups.keys() and type_key in self.valid_types):
                results.append((group_key, type_key))
                
        print("Output: --interpreted", len(results), f"{self.keyword} groups")
        return results
    
    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:

        # print("======= Aggregating =======")
        # Creates an empty dict
        per_link_votes = defaultdict(list)

        # Adds for each answer, its classification for each link
        for model_votes in votes:
            for link, sensor_type in model_votes:
                per_link_votes[link].append(sensor_type)
                
        final = {}
        for link, types in per_link_votes.items():
            # to be classified, the number of votes must be higher than half the number of models
            # if(len(types) > int(self.nb_runs/2)):
            #     final[link] = Counter(types).most_common(1)[0][0]
            most_common, count = Counter(types).most_common(1)[0]
            if count >= int(self.nb_runs/2) + 1:
                final[link] = most_common
        
        print("Validated ", len(final), f"{self.keyword} groups")
        return final


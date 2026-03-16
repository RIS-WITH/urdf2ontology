from tasks.VotingTask import VotingTask

from typing import Dict, Tuple, List, Any
import json
from collections import defaultdict, Counter


# change to Attachment Root Detection
class LinkMountingPointTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
        {
            "chains": [
                {
                "target_link": "<target_link>",
                "attachment_root": "<attachment_root>"
                }
            ]
        }
        """
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "chains": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "link_name": {"type": "string"},
                    "attachment_root": {"type": "string"}
                    },
                    "required": ["link_name", "attachment_root"],
                    "additionalProperties": False
                }
                }
            },
            "required": ["chains"],
            "additionalProperties": False
        }

class ComponentLinkMountingPointTask(LinkMountingPointTask):

    def __init__(self, nb_runs: int, keyword: str):
        super().__init__(nb_runs, keyword)

        self.initial_chains = dict()
                            
    def system_prompt(self) -> str:
        compacted_prompt = f""" 
        You are an expert in ontology abstraction from URDF files.
        You task is to identify the attachment root of {self.keyword} component in their kinematic chain.
        The attachment root is defined as the structural boundary which allows to split the {self.keyword} component from the robot's body.
        For each of the provided target link and their kinematic chain from leaf to root, output the attachment root link. 
        
        Answer only in the following format: {self.json_format}
        For each input chain, produce one entry in "chains".
        Respond ONLY with valid JSON following this structure.
        Do not include explanations.
        """
        
        return compacted_prompt
        
    def user_prompt(self, user_input) -> str:

        chains = json.dumps(user_input, indent=2)

        self.initial_chains = user_input
        
        new_message = f"""
        Below are the identified {self.keyword} target links and their full kinematic chains from the robot base to each leaf links.
        
        Input data: {chains}
        """
        return new_message
    
    def interpret(self, output: Dict[str, Any]) -> List[Tuple[str, str]]:

        valid_results: List[Tuple[str, str]] = []
        sensor_chains = output.get("chains")

        if not isinstance(sensor_chains, list):
            return valid_results
        
        for chain_item in sensor_chains:
            if not isinstance(chain_item, dict):
                continue

            link_name = chain_item.get("target_link")
            mount_point = chain_item.get('attachment_root')
            
            if not isinstance(link_name, str) or not isinstance(mount_point, str):
                continue

            if(link_name not in self.initial_chains.keys()):
                print("The generated link name is non-existing: ", link_name)
                continue
            
            if(mount_point not in self.initial_chains[link_name]): 
                print("The generated attachment root is non-existing: ", mount_point)
                continue
    
            valid_results.append((link_name, mount_point))
        
        print("Interpreted", len(valid_results), f"{self.keyword} groups.")
        return valid_results

    def aggregate(self, votes: List[List[Tuple[str, str, str]]]) -> Dict[str, List[str]]:

        per_link_votes: Dict[str, List[List[str]]] = defaultdict(list)
        final: Dict[str, str] = {}

        for model_votes in votes:
            for link, mount in model_votes:
                per_link_votes[link].append(mount)

        for link, mount in per_link_votes.items():
            most_common, count = Counter(mount).most_common(1)[0]
            if count >= int(self.nb_runs/2) + 1:
                final[link] = most_common
                
        print("Validated", len(final), f"{self.keyword} groups")
        
        return final

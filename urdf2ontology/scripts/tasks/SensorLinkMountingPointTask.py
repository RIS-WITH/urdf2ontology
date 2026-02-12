from tasks.VotingTask import VotingTask

from typing import Dict, Tuple, List, Any
import json
from collections import defaultdict, Counter


class SensorLinkMountingPointTask(VotingTask):

    def __init__(self, nb_runs):
        self.nb_runs = nb_runs
        self.keyword = "sensor"
       
        self.json_format = """
        {
            "chains" : [
                {"link_name": string,  "mounting_point": string},
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
                "mounting_point": {"type": "string"}
                },
                "required": ["link_name", "subchain"],
                "additionalProperties": False
            }
            }
        },
        "required": ["chains"],
        "additionalProperties": False
        }
        
        # chains as arrays of string
        self.initial_chains = dict()
                            
    def system_prompt(self) -> str:
    
        simple_prompt = f"""
        You are given ordered link sequences, each corresponding to an identified {self.keyword} leaf link. Each sequence represents the path from the robot base to a single link.
        For each sequence, identify the mounting link, defined as the link such that:
        - links before it belong to the robot structure, and
        - it and all links after it belong to an attached {self.keyword} (assembly or sub-{self.keyword} component).
        
        Input:
        List of link + chain (ordered list from parent to child) identified as embedding a {self.keyword}-related link as its leaf.
        
        Output:
        For each (link + chain) pair, report the mounting point.
        
        Respond only in JSON format as below:
        """ + self.json_format 
        
        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
        # dump (link + chains) for prompt
        links = json.dumps(user_input, indent=2)
        # store (link + chains) for interpet verification
        self.initial_chains = user_input
        
        # [["base_link", "mount_link", "laser_link"], ["robot_link", "mount_link", "gripper_link"]]
        message = f"Here is the list of {self.keyword}s links that have been identified, along with their hierarchical chain up to the root:" + links
        return message
    
    def interpret(self, output: Dict[str, Any]) -> List[Tuple[str, str]]:

        valid_results: List[Tuple[str, str]] = []
        sensor_chains = output.get("chains")

        if not isinstance(sensor_chains, list):
            return valid_results
        
        for chain_item in sensor_chains:
            # Defensive parsing
            if not isinstance(chain_item, dict):
                continue

            link_name = chain_item.get("link_name")
            mount_point = chain_item.get('mounting_point')
            
            if not isinstance(link_name, str) or not isinstance(mount_point, str):
                continue

            # Check link_name link exists
            initial_chain = self.initial_chains.get(link_name)
            
            if(link_name not in self.initial_chains.keys()):
                print("The generated link name is non-existing: ", link_name)
                continue
            
            if(mount_point not in self.initial_chains[link_name]): 
                print("The generated mount link is non-existing: ", mount_point)
                continue
    
            valid_results.append((link_name, mount_point))
        
        print("Interpreted", len(valid_results), f"{self.keyword} groups.")
        return valid_results
    
    # Perfom the majority vote for each link
    def aggregate(self, votes: List[List[Tuple[str, str, str]]]) -> Dict[str, List[str]]:

        # Collect subchains per link
        per_link_votes: Dict[str, List[List[str]]] = defaultdict(list)
        final: Dict[str, str] = {}

        for model_votes in votes:
            for link, mount in model_votes:
                per_link_votes[link].append(mount)

        for link, mount in per_link_votes.items():
            # to be classified, the number of votes must be higher than half the number of models
            if(len(mount) >= int(self.nb_runs/2) + 1):
                final[link] = Counter(mount).most_common(1)[0][0]
                
        print("Validated", len(final), f"{self.keyword} groups")
        
        return final

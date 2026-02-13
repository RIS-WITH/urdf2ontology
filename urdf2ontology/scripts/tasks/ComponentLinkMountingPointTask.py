from tasks.VotingTask import VotingTask

from typing import Dict, Tuple, List, Any
import json
from collections import defaultdict, Counter

class LinkMountingPointTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
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
                    "required": ["link_name", "mounting_point"],
                    "additionalProperties": False
                }
                }
            },
            "required": ["chains"],
            "additionalProperties": False
        }

class ComponentLinkMountingPointTask(LinkMountingPointTask):

    def __init__(self, nb_runs: int, keyword: str, definition: str):
        super().__init__(nb_runs, keyword)
        self.definition = definition

        self.initial_chains = dict()
                            
    def system_prompt(self) -> str:
    
        old_prompt = f"""
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
        
        new_prompt = f"""
        You are given ordered link sequences.
        Each sequence corresponds to one identified {self.keyword} leaf link.
        Each sequence represents the path from the robot base to that leaf link.

        Your task:
        For each sequence, identify the mounting link.

        Definition of mounting link:
        It is the FIRST link in the chain that belongs to the {self.keyword} assembly.

        More precisely:
        - All links BEFORE the mounting link belong to the robot's structural body.
        - The mounting link and ALL links AFTER it belong to the attached {self.keyword} component.
        - There must be exactly ONE mounting link per chain.

        Selection rules:
        - Choose the earliest link in the chain that clearly belongs to the component.
        - Do NOT choose intermediate structural links.
        - Do NOT choose the base link unless the component is directly attached to it.
        - If uncertain, choose the earliest plausible boundary.

        Respond only in JSON format:
        {self.json_format}
        """
        
        return new_prompt
        
    def user_prompt(self, user_input) -> str:
        # dump (link + chains) for prompt
        links = json.dumps(user_input, indent=2)
        # store (link + chains) for interpet verification
        self.initial_chains = user_input
        
        old_message = f"Here is the list of {self.keyword}s links that have been identified, along with their hierarchical chain up to the root:" + links
        
        new_message = f"""
        Below are the identified {self.keyword} leaf links,
        along with their full kinematic chain from the robot base to the leaf.

        Each entry is:
        - link_name: the identified component leaf link
        - chain: ordered list from base -> ... -> leaf

        Your task:
        For each chain, determine the mounting link.

        Chains: {links}
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

            link_name = chain_item.get("link_name")
            mount_point = chain_item.get('mounting_point')
            
            if not isinstance(link_name, str) or not isinstance(mount_point, str):
                continue

            if(link_name not in self.initial_chains.keys()):
                print("The generated link name is non-existing: ", link_name)
                continue
            
            if(mount_point not in self.initial_chains[link_name]): 
                print("The generated mount link is non-existing: ", mount_point)
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
            # to be classified, the number of votes must be higher than half the number of models
            # if(len(mount) > int(self.nb_runs/2) ):
            #     final[link] = Counter(mount).most_common(1)[0][0]
            most_common, count = Counter(mount).most_common(1)[0]
            if count >= int(self.nb_runs/2) + 1:
                final[link] = most_common
                
        print("Validated", len(final), f"{self.keyword} groups")
        
        return final

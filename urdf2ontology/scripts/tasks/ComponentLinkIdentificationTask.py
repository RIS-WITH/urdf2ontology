from tasks.VotingTask import VotingTask

from typing import Dict, List, Tuple
import json

class LinkIdentificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = f"""
        {{
            "{self.keyword}":
            [
                {{"link_name": "string"}}
            ]
        }}
        """
        
        self.structured_format = {
            "type": "object",
            "properties": {
                self.keyword: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "link_name": {"type": "string"}
                    },
                    "required": ["link_name"],
                    "additionalProperties": False
                }
                }
            },
            "required": [self.keyword],
            "additionalProperties": False
        }

class ComponentLinkIdentificationTask(LinkIdentificationTask):

    def __init__(self, nb_runs: int, keyword: str, definition: str):
        super().__init__(nb_runs, keyword)
        self.definition = definition
                
    def system_prompt(self) -> str:
        simple_prompt = f"""
Task: Identify all URDF leaf links that represent or belong to a {self.keyword} instance.

Definition of {self.keyword}:
{self.definition}

Selection Criteria:
1. Functional Origins: Links representing the point of {self.keyword} interaction.
2. Internal Assemblies: Frames belonging to the internal structure of the {self.keyword}.
3. Exclusions: Do NOT select links that are purely structural or part of the robot's transmission/body.

Input: List of URDF leaf links.
Output Format: Respond strictly in JSON: {self.json_format}
"""

        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
        # store the initial links for verification purposes
        self.init_links = user_input["links"]
        
        # dump in string for prompt
        links = json.dumps(self.init_links, indent=2)
        
        message = f"""
        Below are the URDF leaf links of the robot.
        
        Input data: 
        {links}"""

        return message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        results = []
        sensors = output.get(self.keyword, [])

        if not isinstance(sensors, list):
            return results

        for item in sensors:
            link = item.get("link_name")
            if(link in self.init_links):
                results.append(link)

        print("Interpreted", len(results), self.keyword + "s")
        return results
    
    def aggregate(self, votes: List[List[str]]) -> Dict[str, List]:

        per_link_votes = dict.fromkeys(self.init_links, 0)

        for model_votes in votes:
            for link in model_votes:
                per_link_votes[link] += 1
                
        final = []
        for link, nb_occ in per_link_votes.items():
            if(int(nb_occ) >= int(self.nb_runs/2) + 1):
                final.append(link)
                
        print("Validated ", len(final), self.keyword + "s")
        return final


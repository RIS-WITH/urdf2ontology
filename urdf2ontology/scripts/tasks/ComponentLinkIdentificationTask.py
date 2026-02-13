from tasks.VotingTask import VotingTask

from typing import Dict, List, Tuple
import json

class LinkIdentificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
        {
            """ + f"{self.keyword}" + """: 
            [
                {"link_name": "string"},
            ]
        }
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
        
        # more conservative prompt (does not classify laser_tilt_link, projector_wg6802418_child_frame or l/r_gripper_led_frame )
        simple_prompt = f"""
        You are given URDF leaf links. Your task is to identify all links corresponding to {self.keyword}s.
        
        Definition of {self.keyword}:
        {self.definition}
        
        Include:
        - Direct physical representations.
        - Frames specific to the component.
        - Mounting or tool frames if clearly part of the assembly.

        Input:
        A list of URDF leaf links.

        Output:
        Return only the URDF links identified as {self.keyword}s.
        
        Output Format:
        Respond only with JSON in this format:
        {self.json_format}
        """
        
        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
        # store the initial links for verification purposes
        self.init_links = user_input["links"]

        # dump in string for prompt
        links = json.dumps(self.init_links, indent=2)
        
        message =  f"""
        You are given all URDF leaf links of a robot model.
        A leaf link is a link that has no children in the kinematic chain.

        Your task:
        Identify which of these links correspond to {self.keyword}s.

        Only select links that clearly satisfy the definition.
        Do not guess.
        Do not include unrelated structural parts.
        If unsure, exclude the link.

        Leaf links:{links}
        """

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
            # to be classified, the number of votes must be higher than half the number of models
            if(int(nb_occ) >= int(self.nb_runs/2) + 1):
                final.append(link)
                
        print("Validated ", len(final), self.keyword + "s")
        return final


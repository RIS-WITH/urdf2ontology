from tasks.VotingTask import VotingTask

from typing import Dict, List, Tuple
import json

class SensorLinkIdentificationTask(VotingTask):

    def __init__(self, nb_runs: int):
        self.nb_runs = nb_runs
        self.keyword = "sensor"
        self.keyword_definition = ""
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
        
                  
    def system_prompt(self) -> str:
        
        task_prompt = "You are given a list of leaf links contained in an URDF file describing a robotic system. Your task is to identify only the links that correspond to sensors, discard the ones that do not represent sensors. Respond only with a JSON array following the structure below" + self.json_format
        
        task_prompt = """
        You are an assistant tasked with building an ontology from a robot's URDF. Your goal is to identify all sensor links from the robot model.
        A sensor component may consist of one or more URDF links that together form a single physical sensor assembly, mounted at a common attachment point on the robot.

        Input:
        A list of URDF links corresponding to all of the robot's leaf links.

        Output:
        A list of the URDF links that have been identified as corresponding to a sensor (or part of a sensor assembly).
        Respond only with a JSON array  Respond only with a JSON array following the structure below
        """ + self.json_format
        
        task_prompt = """
        You are an assistant tasked with constructing an ontology from a robot's URDF. Your objective is to identify all URDF links that correspond to sensors.
        A sensor may be represented by:
        - A single URDF link, or
        - Multiple URDF links that together form one physical sensor assembly, all attached at a common mounting point on the robot.

        Input:
        A list of URDF links corresponding to all leaf links in the robot model.

        Task:
        Determine which of the provided leaf links are part of a sensor or sensor assembly.
        If a sensor consists of multiple links, include all links that belong to that sensor assembly.

        Output:
        Return only the URDF links identified as sensors (or parts of sensor assemblies).
        Do not include non-sensor links.
        Do not include explanations, comments, or additional text.

        Output Format:
        Respond only with a JSON object in the following structure:""" + self.json_format
        
        simple_prompt = f"""
        From the URDF leaf links provided, identify all links corresponding to {self.keyword}s. Include links that directly represent {self.keyword}s as well as {self.keyword}-specific frames.
        
        Input:
        A list of URDF links corresponding to all leaf links in the robot model.
        
        Output:
        Return only the URDF links identified as {self.keyword}s (or parts of {self.keyword} assemblies).
        
        Output Format:
        Respond only with a JSON object in the following structure: """ + self.json_format 
        
        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
        # store the initial links for verification purposes
        self.init_links = user_input["links"]

        # dump in string for prompt
        links = json.dumps(self.init_links, indent=2)
        
        message = "Here is the list of leaf links:\n" + links
        return message
    
    # Process the model's outputs to verify that they fit the expected patterns, and create the (link_name) votes
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
    
    # Perfom the majority vote for each link
    def aggregate(self, votes: List[List[str]]) -> Dict[str, List]:

        # Creates an empty dict
        per_link_votes = dict.fromkeys(self.init_links, 0)

        # Adds for each answer, its classification for each link
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


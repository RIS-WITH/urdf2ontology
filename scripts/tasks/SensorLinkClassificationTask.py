from VotingTask import *

import json
from typing import Tuple, List, Dict
from collections import Counter, defaultdict

class SensorLinkClassificationTask(VotingTask):

    def __init__(self, sensor_types: set[str], nb_runs: int):
        self.nb_runs = nb_runs
        self.sensor_types = sensor_types
        
        # self.json_format = """
        # {
        #     "sensors": 
        #     [
        #         {"link_name": "string", "link_type": "one of the predefined types"},
        #     ]
        # }
        # """
        
        self.json_format = """
        {
            "sensors": 
            [
                {"sensor_group": "string", "sensor_type": "one of the predefined types"},
            ]
        }
        """
        
        self.structured_format = {
        "type": "object",
        "properties": {
            "sensors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "link_name": {"type": "string"},
                "link_type": {"type": "string"}
                },
                "required": ["link_name", "link_type"],
                "additionalProperties": False
            }
            }
        },
        "required": ["sensors"],
        "additionalProperties": False
        }
        
        
                         
    def system_prompt(self) -> str:
        
        types = ", ".join(sorted(self.sensor_types))
        # Single link version
        # task_prompt = "You are given a list of links contained in an URDF file describing a robotic system. Your task is to identify only the links that correspond to sensors defined in" + types + ", discard the ones that do not match those classes. Respond with a JSON array that indicates the sensor's link name and its sensor type, following the structure below" + self.json_format
        
        # Sensor group version
        task_prompt = "You are given a list of sensor groups contained in an URDF file describing a robotic system. Each of those sensor groups contains one or several paths to leaf links from a mounting point. Your task is to type those sensor groups based on the type of sensor they represent, based on the predefined types in " + types + ". Respond only with a JSON array that indicates the sensor group's type, following the structure below" + self.json_format
        
        return task_prompt
        
    def user_prompt(self, user_input) -> str:
        # links = json.dumps(user_input["links"], indent=2)
        # message = "Here is the list of links to classify:\n" + links
        
        self.initial_groups = user_input
        groups = json.dumps(user_input, indent=2)
        message = "Here is the list of sensor groups to classify:\n" + groups
        
        return message
    
    # Process the model's outputs to verify that they fit the expected patterns, and create the (link_name, sensor_types) votes
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        # print("======= Interpreting =======")
        results = []

        sensors = output.get("sensors", [])
        if not isinstance(sensors, list):
            return results

        for item in sensors:
            link = item.get("sensor_group")
            sensor_type = item.get("sensor_type")
            # check that the link is a str and that the classified sensor type exists in the valid ontology classes
            if(link in self.initial_groups.keys() and sensor_type in self.sensor_types):
                results.append((link, sensor_type))
                
        # print("Output: --interpreted", len(results), "sensors")
        return results
    
    # Perfom the majority vote for each link
    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:

        # print("======= Aggregating =======")
        # Creates an empty dict
        per_link_votes = defaultdict(list)

        # Adds for each answer, its classification for each link
        for model_votes in votes:
            for link, sensor_type in model_votes:
                per_link_votes[link].append(sensor_type)
                
        # ======= for debug, allows to visualize the votes =======
        # print("before voting: ")
        # for link, types in per_link_votes.items():
        #     print("Link [", link, "] -> ",  types)
        
        final = {}
        for link, types in per_link_votes.items():
            # to be classified, the number of votes must be higher than half the number of models
            if(len(types) > int(self.nb_runs/2)):
                final[link] = Counter(types).most_common(1)[0][0]
        # print("Validated ", len(final), "sensors")
        return final


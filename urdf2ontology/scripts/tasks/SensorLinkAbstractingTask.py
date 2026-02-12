from tasks.VotingTask import VotingTask

from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import json

class SensorLinkAbstractingTask(VotingTask):
    def __init__(self, nb_runs):
        self.nb_runs = nb_runs
        self.initial_chains = {}
        self.initial_chains_joined = {}
        self.keyword = "sensor"
        
        self.json_format = """
        { 
            """ + f"abstracted_{self.keyword}" + """ : [ +
                """ + f"{self.keyword}_id" + """: string, "mounting_point": string},
            ]
        }
        """
        
        self.structured_format = {
        "type": "object",
        "properties": {
            f"abstracted_{self.keyword}": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "sensor_id": {"type": "string"},
                "mounting_point": {"type": "string"}
                },
                "required": [f"{self.keyword}_id", "mounting_point"],
                "additionalProperties": False
            }
            }
        },
        "required": [f"abstracted_{self.keyword}"],
        "additionalProperties": False
        }
        
        self.structured_format = {
            "type": "object",
            "properties": {
                "grouped_sensors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "sensor_name": {"type": "string"},
                    "grouped_links": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                    },
                    "required": ["sensor_name", "grouped_links"],
                    "additionalProperties": False
                }
                }
            },
            "required": ["grouped_sensors"],
            "additionalProperties": False
            }
        
    def system_prompt(self) -> str:
        task_prompt = f"""
        Task:
        Identify the lowest mounting link of a {self.keyword} component in a URDF hierarchy. The mounting link will be abstracted as a single link in an ontology. Therefore, it is important that this mounting point does not have any children links that do not belong to the associated {self.keyword}.
        
        Definitions:
        The mounting point is the highest ancestor link that belongs to the robot body and not to the {self.keyword} sub-tree.
        {self.keyword} group: A set of links representing one physical {self.keyword}.
        Ancestor chain: Ordered list of a link's ancestors up to the root.
        Lowest mounting link: The lowest common ancestor of all links in the {self.keyword} group that represents a physical mounting link.

        Rules:
        Select the lowest valid ancestor as the mounting link.
        Do not forget any {self.keyword} group.

        Input:
        A list of identified {self.keyword} groups, with the grouped links and their associated chains (from their highest mounting point to the leaf links).
        
        Output:
        Respond in a JSON format such as below: 
        """ + self.json_format
        
        return task_prompt
    
    def user_prompt(self, user_input) -> str:
    
        candidates = user_input
        self.initial_chains = user_input
        self.sensor_ids = user_input.keys()
        
        user_prompt = f"Here is the list of {self.keyword} groups: " + json.dumps(candidates, indent=2)
        
        return user_prompt
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:

        # print("======= Interpreting =======")
        results = []

        sensors = output.get(f"abstracted_{self.keyword}", [])
        if not isinstance(sensors, list):
            return results

        for item in sensors:
            sensor_id = item.get(f"{self.keyword}_id")
            mounting_point = item.get("mounting_point")
            # check that the link is a str 
            if(not isinstance(sensor_id, str)):
                continue
            
            results.append((sensor_id, mounting_point))
                
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
            if(len(types) >= int(self.nb_runs/2) + 1):
                final[link] = Counter(types).most_common(1)[0][0]
        # print("Validated ", len(final), "sensors")
        return final

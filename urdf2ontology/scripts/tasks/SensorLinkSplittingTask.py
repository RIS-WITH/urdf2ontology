from tasks.VotingTask import VotingTask
from typing import List, Dict, Tuple, Any

import json
from collections import defaultdict, Counter

class SensorLinkSplittingTask(VotingTask):

    def __init__(self, nb_runs):
        self.nb_runs = nb_runs
        self.keyword = "sensor"
        self.json_format = """
        {
            "chains" : [
                {"link_name": string,  "subchain": [string, string, ...]},
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
                "subchain": {
                    "type": "array",
                    "items": {"type": "string"}
                }
                },
                "required": ["link_name", "subchain"],
                "additionalProperties": False
            }
            }
        },
        "required": ["chains"],
        "additionalProperties": False
        }
        
        # name of the links
        self.all_links = []
        # chains as arrays of string
        self.initial_chains = dict()
                            
    def system_prompt(self) -> str:
        
        task_prompt = """You will be provided with a list of chains, each is an ordered list of URDF links from root (robot's initial link) to leaf (an identified sensor). Extract sensor-specific links by identifying the subchain from each sensor's root component to its leaf link, excluding structural robot links, or bodypart links. Return only the sensor-specific links subchain in hierarchical order for each sensor, including the leaf sensor link. Output format: JSON with sensor names as keys and their corresponding sensor-specific link subchains as values."""  + self.json_format #+ json.dumps(self.structured_format)
        
        # task_prompt = """ You will be provided with a list of chains, where each chain is an ordered list of URDF links from the robot root to a leaf link corresponding to an identified sensor.

        # For each chain, extract the sensor-specific subchain:
        # - The sensor-specific subchain starts at the first link that belongs to the sensor's own embodiment or mounting structure.
        # - Exclude robot-structural and body-part links that are not part of the sensor itself.
        # - Include all sensor-internal links from the sensor root to the leaf sensor link, preserving hierarchical order.

        # Output only the extracted sensor-specific subchains.

        # Respond in JSON format, using the sensor leaf link name as the key and the corresponding sensor-specific link subchain (list of link names) as the value.""" + self.json_format 
        
        compacted_prompt = """
        Task:
        You are given sensor subchains extracted from a URDF. For each subchain, remove all robot structural, body, or mounting links, and keep only the contiguous links intrinsic to the sensor itself. Intrinsic sensor links are links that belong exclusively to the sensor device (e.g., internal frames, housings, optics, electronics). Structural or mounting links are links that would remain if the sensor were removed.

        Input:
        A list of sensor subchains (each an ordered list of link names, parent to child).

        Output:
        Return the trimmed subchains in the same order as the input.

        Rules:
        Remove all non-sensor (robot) links.
        Keep only contiguous intrinsic sensor links.
        Do not reorder links.
        Do not merge or split subchains.
        Every input subchain must produce exactly one trimmed subchain.
        
        Respond in JSON format, using the sensor leaf link name as the key and the corresponding sensor-specific link subchain (list of link names) as the value.""" + self.json_format 
        
        simple_prompt = """
        Task:
        You are given sensor-related link subchains extracted from a URDF (each an ordered list of link names, parent -> child).
        For each subchain, identify the lowest link that belongs to the robot and directly mounts the sensor device (the sensor attachment root).
        The mounting point of a sensor is the link that represents the sensor's own physical body and is attached to the robot via a joint. Shared frames that carry multiple sensors are not mounting points.
        Do not remove or trim links.
        Do not infer intrinsic sensor boundaries.

        Output:
        For each input subchain, return the full subchain starting from the identified robot attachment root down to the sensor leaf.

        Constraints:
        - Preserve link order.
        - Every input subchain produces exactly one output subchain.
        - Respond in JSON, using the sensor leaf link as the key as below
        """ + self.json_format 
        
        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
        links = json.dumps(user_input["chains"], indent=2) 
    
        # initialize the dict with all links as keys
        self.initial_chains = dict.fromkeys(user_input["chains"].keys(), [])

        # store every link and chain for interpret checking
        for link_name, chain in user_input["chains"].items():
            self.initial_chains[link_name] = chain
        
        # [["base_link", "mount_link", "laser_link"], ["robot_link", "mount_link", "gripper_link"]]
        message = f"Here is the list of {self.keyword}s links that have been identified, along with their hierarchical chain up to the root:\n" + links
        
        return message
    
    def interpret(self, output: Dict[str, Any]) -> List[Tuple[str, List[str]]]:

        valid_results: List[Tuple[str, List[str]]] = []
        sensor_chains = output.get("chains")

        if not isinstance(sensor_chains, list):
            return valid_results
        
        for chain_item in sensor_chains:
            # Defensive parsing
            if not isinstance(chain_item, dict):
                continue

            link_name = chain_item.get("link_name")
            subchain = chain_item.get("subchain")
            joined_subchain = ",".join(subchain)
            
            if not isinstance(link_name, str) or not isinstance(subchain, list):
                continue

            # Check leaf link exists
            initial_chain = self.initial_chains.get(link_name)
            joined_initial_chain = ",".join(initial_chain)
            
            if initial_chain is None:
                print("discarded chain because of non-existing link as the leaf:", link_name)
                continue
            elif(len(subchain) == len(initial_chain)):
                print("discarded chain because it is the same length as the initial one.")
                continue
            elif(joined_subchain not in joined_initial_chain):
                print("discarded subchain because it is not contained in the initial chain:\n-initial chain:", joined_initial_chain, "\n-subchain:", joined_subchain)
                continue
            
            # Validate subchain
            for link in subchain:
               if link not in initial_chain:
                    print("discarded chain because of non-existing link in the chain:", link)
                    continue

            valid_results.append((link_name, subchain))

        print("Interpreted", len(valid_results), f"{self.keyword} groups.")
        return valid_results
    
    # Perfom the majority vote for each link
    def aggregate(self, votes: List[List[Tuple[str, List[str]]]]) -> Dict[str, List[str]]:

        # Collect subchains per link
        per_link_votes: Dict[str, List[List[str]]] = defaultdict(list)
        final: Dict[str, List[str]] = {}

        for model_votes in votes:
            for link, subchain in model_votes:
                per_link_votes[link].append(subchain)

        for link, chains in per_link_votes.items():
            if not chains:
                continue

            # Majority vote on chain length (instead of content)
            length_counter = Counter(len(chain) for chain in chains)
            majority_length, _ = length_counter.most_common(1)[0]

            # Select the first chain with the majority length
            for chain in chains:
                if len(chain) == majority_length:
                    final[link] = chain
                    break

        print("Validated", len(final), f"{self.keyword} groups")

        return final

from tasks.VotingTask import VotingTask

from typing import List
import json
from collections import Counter

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
        
    def ollama_json_schema(self):
        return self.structured_format

class ComponentLinkIdentificationTask(LinkIdentificationTask):

    def __init__(self, nb_runs: int, keyword: str):
        super().__init__(nb_runs, keyword)
                
    def system_prompt(self) -> str:
        simple_prompt = f"""
You are an expert in robot kinematics and URDF-to-Ontology abstraction.
Task: You will be provided with a list of links coming from an URDF model. Each link is this list is an URDF leaf link. 
Identify all URDF leaf links that represent or belong to a {self.keyword} instance.

Selection Criteria:
1. Functional Origins: Links representing the point of {self.keyword} interaction.
2. Internal Assemblies: Frames belonging to the internal structure of the {self.keyword}.

Input Format: 
List of URDF leaf links: ["leaf_1", "leaf_2", ...]

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
Do not include any extra text or markdown formatting.
"""

        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
      
        self.init_links = user_input["links"]  # store the initial links for verification purposes
        links = json.dumps(self.init_links, indent=2)
        
        message = f"""
Below are the URDF leaf links of the robot.

Input data: 
{links}"""

        return message
    
    def interpret(self, output: dict) -> List[str]:
        results = []
        sensors = output.get(self.keyword, [])

        if not isinstance(sensors, list):
            print(f"[ERROR] Expected JSON array for key '{self.keyword}', got {type(sensors)}")
            return results

        for item in sensors:
            if not isinstance(item, dict):
                continue
                
            link = item.get("link_name")
            
            if link and link in self.init_links:
                results.append(link)
            elif link:
                print(f"[WARNING] Model guessed non-existent link: '{link}'")

        print(f"Interpreted {len(results)} {self.keyword}s.")
        return results
    
    def aggregate(self, votes: List[List[str]]) -> List[str]:
        vote_counter = Counter()

        for model_run_votes in votes:
            for link in model_run_votes:
                vote_counter[link] += 1
        
        majority_threshold = (len(votes) // 2) + 1
        
        final_validated_links = []
        for link, nb_occ in vote_counter.items():
            if nb_occ >= majority_threshold:
                final_validated_links.append(link)
                
        print(f"Validated {len(final_validated_links)} {self.keyword}s via consensus (Threshold: >= {majority_threshold}/{len(votes)} votes).")
        return final_validated_links


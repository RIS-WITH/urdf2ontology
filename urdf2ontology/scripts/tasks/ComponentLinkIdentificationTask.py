from tasks.VotingTask import VotingTask

from typing import List
import json
from collections import Counter

class LinkIdentificationTask(VotingTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = f"""
{{
    "identifications":
    {{
        "{self.keyword}_0": {{"link_name": "string", "justification": "string"}}
    }}
}}
"""
        self.structured_format = {
    "type": "object",
    "properties": {
        "identifications": {
            "type": "object",
            "description": "Mapping of identified targets to their link name and justification.",
            "minProperties": 1,  # <--- FORCES AT LEAST 1 ITEM TO BE GENERATED
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "link_name": {"type": "string"},
                    "justification": {"type": "string"}
                },
                "required": ["link_name", "justification"],
                "additionalProperties": False
            }
        }
    },
    "required": ["identifications"],
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
Task: You will be provided with a list of leaf links coming from an URDF model. Identify all URDF leaf links that represent or belong to a {self.keyword} instance, as functional origins, internal assemblies or interaction frames. 

Input Format: 
List of URDF leaf links: ["leaf_1", "leaf_2", ...]

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting.
"""

        return simple_prompt
        
    def user_prompt(self, user_input) -> str:
      
        self.init_links = user_input  # store the initial links for verification purposes
        links = json.dumps(self.init_links, indent=2)
        
        message = f"""
Below are the URDF leaf links of the robot.

Input data: 
{links}"""

        return message
    
    def interpret(self, output: dict) -> List[str]:
        results = []
        
        if not isinstance(output, dict):
            return results

        identifications = output.get("identifications", {})

        if not isinstance(identifications, dict):
            print(f"[ERROR] Expected JSON dictionary for key 'identifications', got {type(identifications)}")
            return results

        for id_key, entry_data in identifications.items():
            if not isinstance(entry_data, dict):
                continue
                
            link = entry_data.get("link_name")
            justification = entry_data.get("justification")
            
            if link and link in self.init_links.values():
                results.append((link, justification))
            elif link:
                print(f"[WARNING] Model guessed non-existent link: '{link}'")

        print(f"Interpreted {len(results)} {self.keyword}s.")
        return results
    
    def aggregate(self, votes: List[List[str]]) -> List[str]:
        vote_counter = Counter()

        for model_run_votes in votes:
            for link, justification in model_run_votes:
                vote_counter[link] += 1
        
        majority_threshold = (len(votes) // 2) + 1
        
        final_validated_links = []
        for link, nb_occ in vote_counter.items():
            if nb_occ >= majority_threshold:
                final_validated_links.append(link)
                
        print(f"Validated {len(final_validated_links)} {self.keyword}s via consensus (Threshold: >= {majority_threshold}/{len(votes)} votes).")
        return final_validated_links


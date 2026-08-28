from tasks.VotingTask import VotingTask
from tasks.URDFLinkTask import URDFLinkTask

from typing import Dict, Tuple, List
import json
from collections import defaultdict, Counter

class LinkMountingPointTask(VotingTask, URDFLinkTask):
    def __init__(self, nb_runs, keyword):
        super().__init__(nb_runs, keyword)
        
        self.json_format = """
{
    "mounting_links": {
        "target_id": {
            "attachment_root": "string",
            "justification": "string"
            }
        }
}
"""

        self.structured_format = {
    "type": "object",
    "properties": {
        "mounting_links": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "attachment_root": {"type": "string"},
                    "justification": {"type": "string"}
                },
                "required": ["attachment_root", "justification"],
                "additionalProperties": False
            }
        }
    },
    "required": ["mounting_links"],
    "additionalProperties": False
}
        
    def ollama_json_schema(self):
        return self.structured_format

class ComponentLinkMountingPointTask(LinkMountingPointTask):

    def __init__(self, nb_runs: int, keyword: str, mode: str = "kinematic_csv"):
        super().__init__(nb_runs, keyword)
        self.mode = mode
        self.initial_chains: Dict[str, List[str]] = {}
                            
    def system_prompt(self) -> str:
        keyword_rules = {
    "body_part": (
        "1. External Parent Exclusion: Select the link directly superior to the limb's root joint that is strictly OUTSIDE the limb assembly.\n"
        "2. Inclusion of Base Links: Ensure structural housings belong INSIDE the limb's sub-chain; do not select them as the root."
    ),
    "sensor": (
        "1. External Host Rule: Select the non-sensor structural link that the sensor package attaches to.\n"
        "2. Include Mounting Hardware: The physical sensor bracket, housing, or adapter plate must remain INSIDE the sensor component sub-tree; the attachment root must be the parent link connected to the robot body above that bracket."
    ),
    "end_effector": (
        "1. Flange Boundary Rule: Select the terminal link of the manipulator arm as the external attachment root.\n"
        "2. Adapter Inclusion: Any tool adapter, base plate, or gripper housing must be grouped INSIDE the end-effector component."
    ),
    "locomotion": (
        "1. Chassis Isolation Rule: Select the central body or chassis link superior to the propulsion/locomotion assembly as the attachment root.\n"
        "2. Suspension & Mount Inclusion: All suspension arms, steering knuckles, and motor mount housings must be grouped INSIDE the locomotion component sub-tree, NOT chosen as the root."
        "2. Propulsion & Articulation Inclusion: All intermediate structural mounts and functional sub-components must be grouped INSIDE the locomotion component sub-tree, NOT chosen as the attachment root."
    )
}
        specific_instruction = keyword_rules.get(self.keyword, "1. Identify the structural interface frame joining this subsystem to the primary core.")

        return f"""You are an expert in robot kinematics and URDF-to-Ontology abstraction.
 
Task: Identify the single "Attachment Root" link for the {self.keyword} components provided in the input data.

Input Data Layout:
Each item in the input data represents a targeted component accompanied by its full kinematic chain sequence, ordered sequentially from the robot's primary root link (index 0) down to the functional leaf link (last index).

Definition:
The Attachment Root is the host link on the main structural morphology of the robot to which the entire {self.keyword} component assembly attaches. 

Strict Rules for Selection:
1. It MUST be a link belonging to the main body/structure of the robot that remains after removing the {self.keyword}.
2. It CANNOT be any link internal to the {self.keyword} assembly itself (including its own mounting plate, housing, or leaf links).
3. Think of it as the "parent node" in the URDF tree where you would unbolt or detach the entire {self.keyword} module from the robot.

Strict Boundary Elimination Rules:
{specific_instruction}
3. The "Unscrew" Test: If you physically unbolted or detached this entire functional {self.keyword} assembly (including its brackets and housings) from the robot, which structural link remains intact on the main body? That link is your Attachment Root.
4. The Assembly Housing Exclusion: The Attachment Root MUST NOT be an internal sub-frame that is functionally part of the component assembly itself, or the last link before the core robot structure still belonging to that {self.keyword} component. 
5. Traversal Direction & Chassis Edge-Case: Do not fall into lexical naming traps. Trace leftward (towards index 0) until you exit the component's internal tree.
6. Primary Root Fallback: If the component is mounted directly to the robot's primary root (index 0), then index 0 itself is the Attachment Root.

Input Data Format:
{self.get_input_prompt(self.mode)}

Output Format:
Respond strictly in JSON following this structure: {self.json_format}
For each justification attribute, provide a single, self-sufficient, clear sentence explaining the reason for your choice.
Do not include any extra text or markdown formatting.
"""


    def user_prompt(self, user_input: dict) -> str:

        self.initial_chains = {}
        data_input = [] 
        
        for link, identified_comp in user_input.items():
            self.initial_chains[link] = identified_comp.link_chain
            kinematic_payload = identified_comp.get_kinematic_chain(self.mode)
            entry_input = {
                "target_id": link, 
                "kinematic_chain": kinematic_payload
            }
            data_input.append(entry_input)
                
        chains = json.dumps(data_input, indent=2)
        
        new_message = f"""
Below are the identified {self.keyword} target links and their full kinematic chains from the robot root link to each leaf link.

Input data: 
{chains}
"""
        return new_message
    
    def interpret(self, output: dict) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []

        if not isinstance(output, dict):
            return results

        mounting_links = output.get("mounting_links", {})
        if not isinstance(mounting_links, dict):
            print(f"[PARSE ERROR] 'mounting_links' missing or malformed for task: {self.keyword}")
            return results
        
        for chain_id, entry in mounting_links.items():
            if not isinstance(entry, dict):
                continue
                
            root_link = entry.get("attachment_root")
            justification = entry.get("justification")
            
            if not chain_id or not root_link:
                continue

            if chain_id in self.initial_chains:
                if root_link in self.initial_chains[chain_id]:
                    results.append((chain_id, root_link))
                else:
                    print(f"[ERROR DROPPED] '{root_link}' does not exist on kinematic trace for '{chain_id}'")
            else:
                print(f"[UNKNOWN TARGET] Model provided out-of-bounds component identity: '{chain_id}'")
                    
        print(f"Interpreted {len(results)} valid {self.keyword} mounting links.")
        return results
    
    def aggregate(self, votes: List[List[Tuple[str, str]]]) -> Dict[str, str]:
        per_chain_votes: Dict[str, List[str]] = defaultdict(list)
        final_consensus: Dict[str, str] = {}

        for model_run in votes:
            for chain_id, root_link in model_run:
                per_chain_votes[chain_id].append(root_link)

        # Categorical Selection (Which node is the anchor?)
        majority_threshold = (len(votes) // 2) + 1 if len(votes) > 1 else 1

        for chain_id, vote_list in per_chain_votes.items():
            most_common_data = Counter(vote_list).most_common(1)
            if not most_common_data:
                continue
                
            winner, count = most_common_data[0]
            final_consensus[chain_id] = winner
            
            if count < majority_threshold:
                print(f"[CONSENSUS WARNING] Weak agreement for component '{chain_id}'. Plurality winner: '{winner}' ({count}/{len(votes)} votes)")

        print(f"Validated {len(final_consensus)} finalized {self.keyword} chain roots.")
        return final_consensus
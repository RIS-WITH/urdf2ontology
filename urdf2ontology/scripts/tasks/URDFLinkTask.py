from dataclasses import dataclass, field

@dataclass
class URDFLinkTask:
    input_link_chain = f"""
Kinematic Chain Input Format:
The kinematic chain is an ordered list of links provided using this array structure: 
["start_link", "intermediate_link", "...", "leaf_link"].
"""
    input_kinematic_dict = f"""
Kinematic Chain Input Format:
The kinematic chain is an ordered list of contiguous elements provided using this dictionary structure:
[
    {{
        "parent": "parent_link_name", 
        "joint": {{
            "name": "joint_name", 
            "type": "fixed/revolute/prismatic/continuous"
        }}, 
        "child": "child_link_name"
    }}
]"""
    input_kinematic_txt = f"""
Kinematic Chain Input Format:
The kinematic chain is an ordered list of contiguous elements in a textual form such as: 
"parent_link --(parent_child_joint_name: fixed)--> child_link", ....
"""
    input_kinematic_csv = f"""
Kinematic Chain Input Format:
Th kinematic chain is an ordered list of contiguous elements provided using this array structure:
[["parent_link", "joint_name", "joint_type", "child_link"], ...].
"""

    def get_input_prompt(self, mode):
        input_format = ""
        if(mode == "links"):
            input_format = self.input_link_chain
        elif(mode == "kinematic_dict"):
            input_format = self.input_kinematic_dict
        elif(mode == "kinematic_txt"):
            input_format = self.input_kinematic_txt
        elif(mode == "kinematic_csv"):
            input_format = self.input_kinematic_csv
        
        return input_format
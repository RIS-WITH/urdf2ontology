import ollama
import json
from tasks import VotingTask
import re

class LLMInterface():
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.tasks = []
        self.preload_model()
        self.ontology_handler = None
        self.urdf_handler = None
    
    def preload_model(self):
        ollama.generate(model=self.model,
                        options={"temperature": self.temperature},
                        stream=False,
                        keep_alive=1)
        print(" ========= model loaded ==========")
            
    # Generic method to run a task with a voting mechanism, based on a single input and several outputs of the model
    def run_task(self, task: VotingTask.VotingTask, user_input: str, verbose : bool):
        outputs = []

        for i in range(0, task.nb_runs):
            print("=========== run n°", i, "==========")
            
            # if(verbose):
            #     print("===========> System:", task.system_prompt())
            #     print("===========> User:", task.user_prompt(user_input))
                
            response = ollama.generate(
                model= self.model, 
                system= task.system_prompt(), 
                prompt=task.user_prompt(user_input),
                # format=task.structured_format,
                options={'temperature': self.temperature},
                keep_alive=1,
                think=True #True/False or "low"/"medium"/"high"
            )
                
            try:
                answer_text = re.sub(r"```(?:json)?", "", response["response"]).strip()
                json_answer = json.loads(answer_text)
                outputs.append(json_answer)
            except Exception:
                print("Error: invalid JSON output")
                
            if(verbose):
                print("===========> Answer", response["response"])
            
        votes = []
        for output in outputs:
            try:
                vote = task.interpret(output)
                if vote is not None:
                    votes.append(vote)
            except Exception:
                continue

        if not votes:
            raise ValueError("No valid votes")

        return task.aggregate(votes)
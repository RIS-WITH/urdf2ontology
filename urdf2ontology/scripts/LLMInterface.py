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

            # try:
            #     answer_text = re.sub(r"```(?:json)?", "", response["response"]).strip()
            #     json_answer = json.loads(answer_text)
            #     outputs.append(json_answer)
            
            #  except Exception:
            #     print("Error: invalid JSON output")
                
                
            raw_response = response["response"]
            if(verbose):
                print("===========> Answer", raw_response)
            json_answer = None

            code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
            
            if code_block_match:
                try:
                    json_answer = json.loads(code_block_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

            if json_answer is None:
                outer_braces = re.search(r"(\{.*\})", raw_response, re.DOTALL)
                if outer_braces:
                    try:
                        json_answer = json.loads(outer_braces.group(1).strip())
                    except json.JSONDecodeError:
                        print(f"[ERROR] Found structural braces but failed parsing content for entry.")

            if json_answer is not None:
                outputs.append(json_answer)
            else:
                print(f"[CRITICAL] Error: Invalid JSON output structure from model raw string.")
                
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
import ollama
import json
from tasks import VotingTask
import re
from typing import Optional
import time

class LLMInterface():
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.tasks = []
        self.preload_model()
        self.ontology_handler = None
        self.urdf_handler = None
        self.metrics_log = []  # <--- Stores timing records
    
    def preload_model(self):
        ollama.generate(model=self.model,
                        options={"temperature": self.temperature},
                        stream=False,
                        keep_alive=1)
        print(" ========= model loaded ==========")
            
    # Generic method to run a task with a voting mechanism, based on a single input and several outputs of the model
    def run_task(self, task: VotingTask.VotingTask, user_input: str, verbose: bool = False, verbose_thinking: bool = False):
        outputs = []
        task_name = task.__class__.__name__
        keyword = getattr(task, 'keyword', 'default')

        for i in range(0, task.nb_runs):
            print("=========== run n°", i, "==========")
            
            # if(verbose):
            #     print("=> System:", task.system_prompt())
            #     print("=> User:", task.user_prompt(user_input))
                
            response = ollama.generate(
                model= self.model, 
                system= task.system_prompt(), 
                prompt=task.user_prompt(user_input),
                #format=task.ollama_json_schema(), # not to use with thinking models as it disables output
                options={'temperature': self.temperature},
                keep_alive=1,
                think="medium" #True/False (all models) or "low"/"medium"/"high" (gpt-oss:20b)
            )
                
            raw_response = response["response"]
            
            # Extract Ollama timings (nanoseconds -> seconds)
            prompt_eval_sec = response.get("prompt_eval_duration", 0) / 1e9
            eval_sec = response.get("eval_duration", 0) / 1e9
            prompt_tokens = response.get("prompt_eval_count", 0)
            eval_tokens = response.get("eval_count", 0)

            print(f"Input Tokens Count: {prompt_tokens} (in {prompt_eval_sec:.3f} s)")
            print(f"Output Tokens Count: {eval_tokens} (in {eval_sec:.3f} s)")

            # Save metrics entry
            self.metrics_log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "keyword": keyword,
                "task_name": task_name,
                "run_number": i,
                "prompt_eval_duration_sec": round(prompt_eval_sec, 4),
                "generation_duration_sec": round(eval_sec, 4),
                "total_duration_sec": round(prompt_eval_sec + eval_sec, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": eval_tokens,
                "task_prompt": task.system_prompt(),
                "user_prompt": task.user_prompt(user_input),
                "generation": response["response"]
            })
            
            if(verbose_thinking):
                try:  
                    print("=> Thinking Output:", response["thinking"])
                except KeyError:
                    print("The 'thinking' mode is not enabled")
               
                
            if(verbose):
                print("=> Response Output:", raw_response)
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
    
    # Generic method to run a task with a batched input to avoid the model's hallucination with input overflow
    def run_batched_task(self, task: VotingTask.VotingTask, user_input: str, batch_size: Optional[int] = None, verbose: bool = False, verbose_thinking: bool = False, ):
    # task has already been setup with nb_runs, keyword and input type
        
        MAX_COMPONENTS_PER_PASS = batch_size  
        all_components = list(user_input.items())
        input_length = len(all_components)
        # result_batched = {}
        # result_batched = []
        result_batched = None
        
        for i in range(0, len(all_components), MAX_COMPONENTS_PER_PASS):
            chunk = dict(all_components[i : i + MAX_COMPONENTS_PER_PASS])
            current_pass = (i // MAX_COMPONENTS_PER_PASS) + 1
            total_passes = (len(all_components) + MAX_COMPONENTS_PER_PASS - 1) // MAX_COMPONENTS_PER_PASS
            
            print(f"--- Processing Sub-Pass {current_pass}/{total_passes} (Batch Size: {len(chunk)}) ---")
            
            chunk_result = self.run_task(task, chunk, verbose=verbose, verbose_thinking=False)
            print(chunk_result)
            # result_batched.update(chunk_result)
            ##result_batched.extend(chunk_result)
            
            # 1. Handle Dictionary outputs (e.g., attachment roots dict)
            if isinstance(chunk_result, dict):
                if result_batched is None:
                    result_batched = {}
                result_batched.update(chunk_result)
                
            # 2. Handle List outputs (e.g., list of link names)
            elif isinstance(chunk_result, list):
                if result_batched is None:
                    result_batched = []
                result_batched.extend(chunk_result)
                
            # 3. Fallback for primitive outputs (strings, ints, etc.)
            else:
                if result_batched is None:
                    result_batched = []
                result_batched.append(chunk_result)

            
        return result_batched
    
    def save_metrics(self, output_path: str):
        """Saves collected timing metrics to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.metrics_log, f, indent=4)
        print(f"📊 Saved timing report to {output_path}")
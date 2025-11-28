import copy
import importlib
import time,os,json,asyncio,re,random
import pandas as pd
import numpy as np
import shutil

from termcolor import colored
from tqdm import tgrange
from pathlib import Path
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
from typing import Dict,Any
from pydantic_core import to_jsonable_python

from .all_prompt import *
from .evaluate import evaluate_math,evaluate_mbpp,extract_model_answer
from ..mas_base import MAS
from ..utils import load_config

BENCHMARK={
    "math": {
        "operators":["Custom", "ScEnsemble", "Programmer"],
        "type":"math"
        },
    "mbpp": {
        "operators":["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
        "type":"code"
        }
}

class AFlow(MAS):
    def __init__(self, general_config, method_config_name="config"):
        super().__init__(general_config)
        
        self.method_config = load_config(
            Path(__file__).parent / "configs" / f"{method_config_name}.yaml"
        )
        self.dataset_name = general_config['test_dataset_name']
        self.model_name_optimize = self.method_config.get('optimize_meta_model_name','gpt-4o') 
        self.model_name_execute = self.method_config.get('optimize_execute_model_name','gpt-4o-mini-2024-07-18')
        self.sample = self.method_config['sample']
        self.max_rounds = self.method_config['max_rounds']
        self.validation_rounds = self.method_config['validation_rounds']
        self.earlystop = self.method_config['earlystop']
        self.root_path = str(os.path.relpath(Path(__file__).parent, start=os.getcwd()))
        self.results_path = f"results/{self.dataset_name}/aflow/{self.model_name_optimize}/{self.model_name_execute}"
        self.top_scores = []
        self.round = 1
        self.graph = None

        matches = re.findall('(math|mbpp)', self.dataset_name.lower(), flags=re.IGNORECASE)
        if matches:     
            self.operators:list = BENCHMARK[matches[0]]["operators"]
            self.type = BENCHMARK[matches[0]]["type"]
            self.domain = matches[0]
        else:
            raise ValueError("Dataset not found!")
        
        results_path = Path(self.results_path)
        if not results_path.exists():
            graph_path = Path(self.root_path) / "initial_workflows" / self.domain
            results_path.mkdir(parents=True, exist_ok=True)
            exp_path = os.path.join(self.results_path, "processed_experience.json")
            res_path = os.path.join(self.results_path, "results.json")
            with open(exp_path, 'w') as f:
                pass
            with open(res_path, 'w') as f:
                pass
            for item in graph_path.iterdir():
                dest = results_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        self.optimized_round = 1
        self.inference_flag = True

    
    def inference(self, query,entrypoint=""):
        """
        query: Query to be passed to the MAS
        """
        self.inference_flag = True
        optimized_path = Path(self.results_path) / "best_workflow"
        if optimized_path.exists():
            graph_module_name = f"results.{self.dataset_name}.aflow.{self.model_name_optimize}.{self.model_name_execute}.best_workflow.graph"
        else:
            raise NotImplementedError("Best_workflow path does not exist!")
        module = importlib.import_module(graph_module_name, package=__package__)
        self.graph = getattr(module, "Workflow")
        
        graph = self.graph(name="Optimized", env=self)
        
        if self.domain == "math":
            response = asyncio.run(graph(problem=query))
            #print("Raw response: ",response)
            response = extract_model_answer(response)
            #print("Porcessed response: ",response)
        else:
            response = asyncio.run(graph(problem=query,entry_point=entrypoint))
        return response

    def optimizing(self,val_dataset):
        self.inference_flag = False
        
        optimized_path = Path(self.results_path) / "best_workflow"
        if optimized_path.exists():
            print(colored("The optimal graph already exists!\n","red"))
            return
    
        print(colored("Start optimizing ...\n","yellow"))
        for i in range(self.max_rounds):
            try:
                print(colored(f"{i+1} round of optimization...\n","light_cyan"))
                score = asyncio.run(self._optimize_graph(val_dataset))
            except Exception as e:
                print(f"Optimization failed: {e}")
                score = None
            self.round += 1
            print(f"Score for round {self.round}: {score}")
            self.save_optimized_graph()
            converged, convergence_round, final_round = self.check_convergence(top_k=3)
            if self.earlystop and converged:
                print(f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}")
                self.print_results()
                break
            time.sleep(5)
        print(colored("Optimization complete!","green"))
        print(colored(f"\n>> Optimization token stats: {self.get_token_stats()}","light_yellow"))

    async def _optimize_graph(self,val_dataset):

        validation_n = self.validation_rounds  
        graph_path = self.results_path
        result_path = os.path.join(graph_path, "results.json")
        data=[]
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        if self.round == 1:
            directory = os.path.join(graph_path, f"round_{self.round}")
            os.makedirs(directory, exist_ok=True)
    
            graph_module_name = f"results.{self.dataset_name}.aflow.{self.model_name_optimize}.{self.model_name_execute}.round_{self.round}.graph"
            module = importlib.import_module(graph_module_name, package=__package__)
            self.graph = getattr(module, "Workflow")
            avg_score = await self.evaluate_graph(directory, validation_n, data,val_dataset,True)

    
        while True:
            directory = os.path.join(graph_path, f"round_{self.round+1}")
            os.makedirs(directory, exist_ok=True)
            
            #parent <- SelectParent(results)
            top_rounds = self.get_top_rounds()
            sample,_ = self.select_round(top_rounds)

            prompt, graph_load = self.read_graph_files(sample["round"], graph_path)
            pattern = r"class Workflow:.+"
            graph = re.findall(pattern, graph_load, re.DOTALL)

            #context <- LoadContext(parent,experiences)
            processed_experience = self.load_experience()
            experience = self.format_experience(processed_experience, sample["round"])

            path = os.path.join(graph_path, "template/operator.json")
            operators_description = ""
            for id, operator in enumerate(self.operators):
                with open(path, "r") as f:
                    operator_data = json.load(f)
                    matched_data = operator_data[operator]
                    desc = matched_data["description"]
                    interface = matched_data["interface"]
                    operator_description = f"{id+1}. {operator}: {desc}, with interface {interface})."
                operators_description += f"{operator_description}\n"

            log_data = self.load_log(sample["round"])

            graph_input = WORKFLOW_INPUT.format(
                experience=experience,
                score=sample["score"],
                graph=graph[0],
                prompt=prompt,
                operator_description=operator_description,
                type=self.type,
                log=log_data,
            )
            graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=self.type)
            graph_optimize_prompt = graph_input + WORKFLOW_CUSTOM_USE + graph_system
            names = ["modification","graph","prompt"]
            types = {"modification":str,"graph":str,"prompt":str}
            examples = []
            for name in names:
                examples.append(f"<{name}>content</{name}>")

            example_str = "\n".join(examples)
            graph_optimize_prompt += f"""
                ### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
                {example_str}
            """
            response = self.call_llm(prompt=graph_optimize_prompt,model_name=self.model_name_optimize)

            response = self.xml_extract(response,names,types)
            # Check if the modification meets the conditions
            check = self.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            # If `check` is True, break the loop; otherwise, regenerate the graph
            if check:
                break

        # Save the graph and evaluate
        graph = WORKFLOW_TEMPLATE.format(graph=response["graph"], round=self.round + 1, dataset=self.dataset_name)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
        experience = {
            "father node": sample["round"],
            "modification": response["modification"],
            "before": sample["score"],
            "after": None,
            "succeed": None,
        }
        graph_module_name = f"results.{self.dataset_name}.aflow.{self.model_name_optimize}.{self.model_name_execute}.round_{self.round+1}.graph"
        module = importlib.import_module(graph_module_name, package=__package__)
        self.graph = getattr(module, "Workflow")
        avg_score = await self.evaluate_graph(directory, validation_n, data,val_dataset)
        
        experience["after"] = avg_score
        experience["succeed"] = bool(avg_score > experience["before"])
        folder_path = Path(os.path.join(directory, "experience.json")).parent
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(directory, "experience.json"), "w", encoding="utf-8") as fout:
            json.dump(experience, fout, ensure_ascii=False, indent=4, default=to_jsonable_python)
        return avg_score
    
    async def evaluate_graph(self, directory, validation_n, data, val_dataset,initial=False):
        sum_score = 0
        max_concurrent_tasks = 50
        for i in range(validation_n):
            graph = self.graph(name=self.dataset_name+f"/round_{self.round}", env=self)
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [self._run_with_semaphore(semaphore, problem,directory,graph)for problem in val_dataset]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.dataset_name} problems", total=len(val_dataset))

            columns = ["question", "prediction", "expected_output", "score"]
            df = pd.DataFrame(results, columns=columns)
            average_score = df["score"].mean()
            cur_round = self.round + 1 if initial is False else self.round
            new_data = {"round": cur_round, "score": average_score}
            data.append(new_data)

            result_path = os.path.join(self.results_path, "results.json")
            folder_path = Path(result_path).parent
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)

            with open(result_path, "w", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False, indent=4, default=to_jsonable_python)
            sum_score += average_score

        return sum_score / validation_n
    
    async def _run_with_semaphore(self, semaphore, problem,log_path,graph):
        async with semaphore:
            if self.domain == "math":
                return await evaluate_math(problem, graph,log_path)
            else:
                return await evaluate_mbpp(problem, graph,log_path)
    
    def check_convergence(self, top_k=3, z=0, consecutive_rounds=5):
        result_file = os.path.join(self.results_path, "results.json")
        with open(result_file, "r") as file:
            self.data = json.load(file)
        rounds = {}
        for entry in self.data:
            round_number = entry["round"]
            score = entry["score"]
            if round_number not in rounds:
                rounds[round_number] = []
            rounds[round_number].append(score)
        self.rounds = rounds
        sorted_rounds = sorted(self.rounds.items(), key=lambda x: x[0])
        avg_scores = []
        stds = []
        for round_number, scores in sorted_rounds:
            avg_scores.append(np.mean(scores))
            stds.append(np.std(scores))
        # If total rounds are not enough to calculate top_k+1 rounds, return not converged
        if len(avg_scores) < top_k + 1:
            return False, None, None
        convergence_count = 0  # Convergence counter
        previous_y = None  # Y value of the previous round (average of top_k scores)
        sigma_y_previous = None  # Standard error of Y value from previous round
        for i in range(len(avg_scores)):
            # Dynamically select top_k from current round and all previous rounds
            top_k_indices = np.argsort(avg_scores[: i + 1])[::-1][:top_k]  # Select top k indices by descending average score
            top_k_scores = [avg_scores[j] for j in top_k_indices]  # Get list of top k scores
            top_k_stds = [
                stds[j] for j in top_k_indices
            ]  # Get list of standard deviations corresponding to top k scores
            # Calculate mean of top k scores for current round, i.e., y_current
            y_current = np.mean(top_k_scores)
            # Calculate standard error of y_current (sigma_y_current), representing score dispersion
            sigma_y_current = np.sqrt(np.sum([s**2 for s in top_k_stds]) / (top_k**2))
            # If not the first round, calculate change in Y (Delta_Y) and corresponding standard error
            if previous_y is not None:
                # Calculate Y difference between current round and previous round
                delta_y = y_current - previous_y
                # Calculate standard error of Y difference (sigma_Delta_Y)
                sigma_delta_y = np.sqrt(sigma_y_current**2 + sigma_y_previous**2)
                # Check if Y change is within acceptable confidence interval, i.e., convergence condition
                if abs(delta_y) <= z * sigma_delta_y:
                    convergence_count += 1
                    # If consecutive converged rounds reach set value, return convergence information
                    if convergence_count >= consecutive_rounds:
                        return True, i - consecutive_rounds + 1, i
                else:
                    # If change is large, reset convergence counter
                    convergence_count = 0
            # Update Y value and standard error for previous round
            previous_y = y_current
            sigma_y_previous = sigma_y_current
        # If convergence condition not met, return not converged
        return False, None, None
    
    def get_top_rounds(self):
        rounds_dir = self.results_path
        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []
        if not Path(result_file).exists():
            raise FileNotFoundError(f"json_file: {result_file} not exist, return []")
        with open(result_file, "r", encoding="utf-8") as fin:
            try:
                data = json.load(fin)
            except Exception:
                raise ValueError(f"read json file: {result_file} failed")
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": int(round_number), "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= self.sample:
                    break

        return unique_top_scores
    
    def select_round(self, items,alpha=0.2, lambda_=0.3):
        
        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        
        print(f"\nMixed probability distribution: {mixed_prob}")
        print(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=mixed_prob)
        print(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index],sorted_items
    
    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            print(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            print(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content
    
    def load_experience(self):
        rounds_dir = os.path.normpath(self.results_path)
        experience_data = defaultdict(lambda: {"score": None, "success": {}, "failure": {}})

        for round_dir in os.listdir(rounds_dir):
            if os.path.isdir(os.path.join(rounds_dir, round_dir)) and round_dir.startswith("round_"):
                round_path = os.path.join(rounds_dir, round_dir)
                try:
                    round_number = int(round_dir.split("_")[1])
                    json_file_path = os.path.join(round_path, "experience.json")
                    if os.path.exists(json_file_path):
                        if not Path(json_file_path).exists():
                            raise FileNotFoundError(f"json_file: {json_file_path} not exist, return []")
                        with open(json_file_path, "r", encoding="utf-8") as fin:
                            try:
                                data = json.load(fin)
                            except Exception:
                                raise ValueError(f"read json file: {json_file_path} failed")
                    
                        father_node = data["father node"]

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = data["before"]

                        if data["succeed"]:
                            experience_data[father_node]["success"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        else:
                            experience_data[father_node]["failure"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                except Exception as e:
                    print(f"Error processing {round_dir}: {str(e)}")

        experience_data = dict(experience_data)

        output_path = os.path.join(rounds_dir, "processed_experience.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(experience_data, outfile, indent=4, ensure_ascii=False)

        print(f"Processed experience data saved to {output_path}")
        return experience_data
    
    def format_experience(self, processed_experience, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            experience = f"Original Score: {experience_data['score']}\n"
            experience += "These are some conclusions drawn from experience:\n\n"
            for key, value in experience_data["failure"].items():
                experience += f"-Absolutely prohibit {value['modification']} (Score: {value['score']})\n"
            for key, value in experience_data["success"].items():
                experience += f"-Absolutely prohibit {value['modification']} \n"
            experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
        else:
            experience = f"No experience data found for round {sample_round}."
        return experience
    
    def load_log(self, cur_round):
        log_dir = os.path.join(self.results_path, f"round_{cur_round}/log.json")
        if not os.path.exists(log_dir):
            return ""  
        print(log_dir)
        if not Path(log_dir).exists():
            raise FileNotFoundError(f"json_file: {log_dir} not exist, return []")
        with open(log_dir, "r", encoding="utf-8") as fin:
            try:
                data = json.load(fin)
            except Exception:
                raise ValueError(f"read json file: {log_dir} failed")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log
    
    def check_modification(self, processed_experience, modification, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            for key, value in experience_data["failure"].items():
                if value["modification"] == modification:
                    return False
            for key, value in experience_data["success"].items():
                if value["modification"] == modification:
                    return False
            return True
        else:
            return True
        
    def print_results(self):
        """
        Print average score and standard deviation for all rounds.
        """
        rounds_dir = os.path.normpath(self.results_path)
        result_file = os.path.join(rounds_dir, "results.json")
        # Ensure directory exists
        os.makedirs(rounds_dir, exist_ok=True)
        # If file doesn't exist, create a new one with an empty list
        if not os.path.exists(result_file):
            with open(result_file, "w") as file:
                json.dump([], file)
        # Read file and return data
        with open(result_file, "r") as file:
            return json.load(file)
        rounds = {}
        for entry in self.data:
            round_number = entry["round"]
            score = entry["score"]
            if round_number not in rounds:
                rounds[round_number] = []
            rounds[round_number].append(score)
        return rounds
        sorted_rounds = sorted(self.rounds.items(), key=lambda x: x[0])
        avg_scores = []
        stds = []
        for round_number, scores in sorted_rounds:
            avg_scores.append(np.mean(scores))
            stds.append(np.std(scores))
        return avg_scores, stds
        for i, (avg_score, std) in enumerate(zip(self.avg_scores, self.stds), 1):
            print(f"Round {i}: Average Score = {avg_score:.4f}, Standard Deviation = {std:.4f}")
    
    def extract(self,response):
        TAG = "CONTENT"
        req_key=f"[/{TAG}]"
            
        def re_extract_content(cont,pattern):
            matches = re.findall(pattern, cont, re.DOTALL)
            for match in matches:
                if match:
                    cont = match
                    break
            return cont.strip()
        raw_content = copy.deepcopy(response)
        pattern = r"\[CONTENT\]([\s\S]*)\[/CONTENT\]"
        new_content = re_extract_content(raw_content, pattern)
        if not new_content.startswith("{"):
        # TODO find a more general pattern
        # # for `[CONTENT]xxx[CONTENT]xxxx[/CONTENT] situation
            print(f"extract_content try another pattern: {pattern}")
            if req_key not in new_content:
                raw_content = copy.deepcopy(new_content + "\n" + req_key)
        # # pattern = r"\[CONTENT\](\s*\{.*?\}\s*)\[/CONTENT\]"
            new_content = re_extract_content(raw_content, pattern)
        else:
            if req_key in new_content:
                idx = new_content.find(req_key)
                new_content = new_content[:idx]
                new_content = new_content.strip()
        return json.JSONDecoder(strict=False).decode(new_content,_w=json.decoder.WHITESPACE.match)
    
    @staticmethod
    def xml_extract(context: str,field_names :list,field_types) -> Dict[str, Any]:
        """
        Fill context with XML tags and convert according to field types, including string, integer, boolean, list and dict types
        """
        extracted_data: Dict[str, Any] = {}

        for field_name in field_names:
            pattern = rf"<{field_name}>(.*?)</{field_name}>"
            match = re.search(pattern, context, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                field_type = field_types.get(field_name)

                if field_type == str:
                    extracted_data[field_name] = raw_value
                elif field_type == int:
                    try:
                        extracted_data[field_name] = int(raw_value)
                    except ValueError:
                        extracted_data[field_name] = 0  
                elif field_type == bool:
                    extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
                elif field_type == list:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], list):
                            raise ValueError
                    except:
                        extracted_data[field_name] = [] 
                elif field_type == dict:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], dict):
                            raise ValueError
                    except:
                        extracted_data[field_name] = {}  

        return extracted_data
    
    def save_optimized_graph(self):
        top_rounds = self.get_top_rounds()
        sample,items = self.select_round(top_rounds)
        graph_path = Path(self.results_path) 
        self.optimized_round=items[0]["round"]

        source_round = graph_path / f"round_{self.optimized_round}"
        dest_round = graph_path / "best_workflow"
        if source_round.exists():
            shutil.copytree(source_round, dest_round, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"The source folder {source_round} does not exist.")
        
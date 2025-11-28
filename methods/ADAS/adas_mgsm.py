import argparse
import copy
import json
import numpy as np
import openai
import os
import random

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from ..mas_base import MAS
from ..utils import load_config
from .adas_utils import random_id, bootstrap_confidence_interval, score_mgsm
from .prompt.mgsm_prompt import get_init_archive, get_prompt, get_reflexion_prompt



Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True
execute_model = None
api_key = "http://47.88.65.188:8405/v1"
base_url = "sk-Hu8QQ1yseCFCMxc209Ab0cF8Fe3e49C4A52b2544F176B1Df" 
optimize_execute_token_stats = {}
inference_execute_token_stats = {}

def merge_token_stats(target, *sources):
    for d in sources:
        for model_key, stats in d.items():
            for k, v in stats.items():
                target.setdefault(model_key, {}).setdefault(k, 0)
                target[model_key][k] += v
    return target

class AgentSystem():
    def __init__(self) -> None:
        pass

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = execute_model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    def get_json_response_from_gpt(
            self,
            msg,
            model,
            system_message,
            temperature=0
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        json_dict = json.loads(content)
        num_prompt_tokens = response.usage.prompt_tokens
        num_completion_tokens = response.usage.completion_tokens
        if isinstance(content, str) and SEARCHING_MODE:       # in cases where response is None or an error message
            if model not in optimize_execute_token_stats:
                optimize_execute_token_stats[model] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            optimize_execute_token_stats[model]["num_llm_calls"] += 1
            optimize_execute_token_stats[model]["prompt_tokens"] += num_prompt_tokens
            optimize_execute_token_stats[model]["completion_tokens"] += num_completion_tokens
        elif isinstance(content, str) and not SEARCHING_MODE:
            if model not in inference_execute_token_stats:
                inference_execute_token_stats[model] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            inference_execute_token_stats[model]["num_llm_calls"] += 1
            inference_execute_token_stats[model]["prompt_tokens"] += num_prompt_tokens
            inference_execute_token_stats[model]["completion_tokens"] += num_completion_tokens
        # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
        assert not json_dict is None
        return json_dict

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}. Return ONLY the number, i.e. 121 or 9." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = self.get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)
    
class ADAS_MGSM(MAS):
    def __init__(self, general_config, method_config_name="config"):
        super().__init__(general_config)
        # set the meta model and execute model for optimizing mode

        self.model_api_config = general_config["model_api_config"]
        self.optimize_execute_model = general_config.get('optimize_execute_model_name','gpt-4o-mini-2024-07-18')
        self.optimize_meta_model = general_config.get('optimize_meta_model_name','gpt-4o')
        global execute_model, api_key, base_url
        execute_model = self.optimize_execute_model
        self.execute_model_dict = self.model_api_config[execute_model]['model_list'][0]
        api_key = self.execute_model_dict['api_key']
        base_url = self.execute_model_dict['model_url']
        self.max_workers = self.model_api_config[execute_model]["max_workers"]        
        self.inference_model = general_config['model_name']

        self.config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml"))
        self.domain = method_config_name
        self.valid_size = self.config["valid_size"]
        self.test_size = self.config["test_size"]
        self.shuffle_seed = self.config["shuffle_seed"]
        self.n_repreat = self.config["n_repreat"]
        self.multiprocessing = self.config["multiprocessing"]
        self.debug = self.config["debug"]

        self.save_dir = Path(__file__).parents[2] / "results" / self.domain / f"adas_{self.optimize_meta_model}_optimize_{self.optimize_execute_model}_execute"
        self.optimizing_path = self.save_dir / "archive.json"
        self.inference_path = self.save_dir / "best_workflow.json"
        self.n_generation = self.config["n_generation"]
        self.debug_max = self.config["debug_max"]
        self.args = argparse.Namespace(**self.config)


    def call_llm(self, prompt=None, system_prompt=None, messages=None):
        response = super().call_llm(prompt=prompt, system_prompt=system_prompt, messages=messages, model_name=self.optimize_meta_model)
        formatted_response = response.strip().replace('```json', '').replace('```', '')
        return formatted_response
        
    def evaluate_forward_fn(self, forward_str, val_dataset, searching_mode=True):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
        namespace = {}
        exec(forward_str, globals(), namespace)
        names = list(namespace.keys())
        if len(names) != 1:
            raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
        func = namespace[names[0]]
        if not callable(func):
            raise AssertionError(f"{func} is not callable")
        setattr(AgentSystem, "forward", func)
        # print(f"forward function defined:\n{forward_str}")

        # set seed 0 for valid set
        # examples = get_all_examples()
        random.seed(self.shuffle_seed)
        examples = random.sample(val_dataset, len(val_dataset))
        random.shuffle(examples)

        if searching_mode:
            examples = examples[:self.valid_size] * self.n_repreat
        else:
            examples = examples[self.valid_size:self.valid_size + self.test_size] * self.n_repreat

        questions = ['Solve this math problem.\n' + example['query'] for example in examples]
        answers = [example['answer_number'] for example in examples]

        print(f"problem length: {len(examples)}")
        max_workers = min(len(examples), self.max_workers) if self.multiprocessing else 1

        task_queue = []
        for q in questions:
            taskInfo = Info('task', 'User', q, -1)
            task_queue.append(taskInfo)

        agentSystem = AgentSystem()

        acc_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

        for q_idx, res in enumerate(results):
            try:
                if isinstance(res, Info):
                    extracted_answer = res.content
                else:
                    extracted_answer = res
                extracted_answer = int(extracted_answer.strip())
                correct_answer = answers[q_idx]
                correct = bool(extracted_answer == correct_answer)
                # correct = score_mgsm(correct_answer, extracted_answer)
            except Exception as e:
                acc_list.append(0)
                continue

            acc_list.append(1 if correct else 0)

        print(f"acc: {bootstrap_confidence_interval(acc_list)}")
        return acc_list

    def optimizing(self, val_dataset):
        # create save dir
        os.makedirs(os.path.dirname(self.optimizing_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.inference_path), exist_ok=True)

        best_fitness = 0
        # training 
        # use validation dataset to optimize
        archive = get_init_archive()
        start = 0

        for solution in archive:
            if 'fitness' in solution:
                continue

            solution['generation'] = "initial"
            print(f"============Initial Archive: {solution['name']}=================")
            # self.dynamic_forward(solution["code"])
            try:    
                acc_list = self.evaluate_forward_fn(solution["code"],val_dataset)
            except Exception as e:
                print("During evaluating initial archive:")
                print(e)
                continue
            
            fitness, fitness_str = bootstrap_confidence_interval(acc_list)
            solution['fitness'] = fitness_str
            # save the agent with the highest median score
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

            with open(self.optimizing_path, 'w') as json_file:
                json.dump(archive, json_file, indent=4)

        for n in range(start, self.n_generation):
            print(f"============Generation {n + 1}=================")
            system_prompt, prompt = get_prompt(archive)
            msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            ]

            try:
                next_solution = self.call_llm(messages=msg_list)
                print(next_solution)
                Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
                # Reflexion 1
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_1})
                next_solution = self.call_llm(messages=msg_list)

                # Reflexion 2
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_2})
                next_solution = self.call_llm(messages=msg_list)
                next_solution = next_solution.strip().replace('```json', '').replace('```', ',')
                next_solution = json.loads(next_solution)
            except Exception as e:
                print("During optimizing:")
                print(e)
                n -= 1 
                continue
            
            acc_list = []
            for _ in range(self.debug_max):
                try:
                    acc_list = self.evaluate_forward_fn(next_solution["code"], val_dataset)
                    if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                        raise Exception("All 0 accuracy")
                    break
                except Exception as e:
                    print("During evaluation:")
                    print(e)
                    msg_list.append({"role": "assistant", "content": str(next_solution)})
                    msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                    try:
                        next_solution = self.call_llm(msg_list)
                    except Exception as e:
                        print("During LLM generate new solution:")
                        print(e)
                        continue
                    continue
            if not acc_list:
                n -= 1
                continue

            fitness, fitness_str = bootstrap_confidence_interval(acc_list)
            next_solution['fitness'] = fitness_str
            next_solution['generation'] = n + 1

            if 'debug_thought' in next_solution:
                del next_solution['debug_thought']
            if 'reflection' in next_solution:
                del next_solution['reflection']
            print(next_solution['name'])
            print(next_solution['code'])

            # save the agent with the highest median score 
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = next_solution

            archive.append(next_solution)

            with open(self.optimizing_path, 'w') as json_file:
                json.dump(archive, json_file, indent=4)

        with open(self.inference_path, 'w') as json_file:
            json.dump(best_solution, json_file, indent=4)
    
    def inference(self, query):
        global execute_model, api_key, base_url, SEARCHING_MODE
        SEARCHING_MODE = False
        execute_model = self.inference_model
        self.execute_model_dict = self.model_api_config[execute_model]['model_list'][0]
        api_key = self.execute_model_dict['api_key']
        base_url = self.execute_model_dict['model_url']
        self.max_workers = self.model_api_config[execute_model]["max_workers"]
        if not os.path.exists(self.inference_path):
            raise NotImplementedError("The specified best workflow path does not exist.")
        with open(self.inference_path, 'r') as json_file:
            best_solution = json.load(json_file)
        
        namespace = {}
        exec(best_solution["code"], globals(), namespace)
        names = list(namespace.keys())
        if len(names) != 1:
            raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
        func = namespace[names[0]]

        if not callable(func):
            raise AssertionError(f"{func} is not callable")
        setattr(AgentSystem, "forward", func)
        agentSystem = AgentSystem()

        taskInfo = Info('task', 'User', query, -1)
        response = agentSystem.forward(taskInfo)
        if isinstance(response, str):
            response = response
        else:
            response = response.content

        merge_token_stats(self.token_stats, optimize_execute_token_stats, inference_execute_token_stats)

        return response
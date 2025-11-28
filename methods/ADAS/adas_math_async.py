import argparse
import asyncio
import copy
import json
import numpy as np
import openai
import os
import random
import time

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

from ..mas_base import MAS
from ..utils import load_config
from .adas_utils import random_id, bootstrap_confidence_interval
from .prompt.main_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from .evaluate_math_aflow import grade_answer

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""
optimize_execute_token_stats = {}
inference_execute_token_stats = {}
PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

client = openai.OpenAI(api_key='sk-lDzVviBBxEvzIk9Ay4oFPLwLEAqtSZqxxdgaEWAX0Nl9FXxm', base_url='http://123.129.219.111:3000/v1')

def get_json_response_from_gpt(msg,model,system_message,temperature=0):
    response = client.chat.completions.create(
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

class AgentSystem():
    def __init__(self) -> None:
        pass

class LLMAgentBase():
    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-4o-mini-2024-07-18', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
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
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
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
    
class ADAS_MATH(MAS):
    def __init__(self, general_config, method_config_name="config"):
        super().__init__(general_config)
        self.method_config = load_config(
            Path(__file__).parent / "configs" / f"{method_config_name}.yaml"
        )
        self.dataset_name = general_config['test_dataset_name']
        self.model_name_optimize = self.method_config.get('optimize_meta_model_name','gpt-4o-2024-08-06') 
        self.model_name_execute = self.method_config.get('optimize_execute_model_name','gpt-4o-mini-2024-07-18')
        global execute_model, api_key, base_url
        execute_model = self.model_name_execute
        self.execute_model_dict = self.model_api_config[execute_model]['model_list'][0]
        api_key = self.execute_model_dict['api_key']
        base_url = self.execute_model_dict['model_url']
        self.inference_model = general_config['model_name']

        self.valid_size = self.method_config["valid_size"]
        self.test_size = self.method_config["test_size"]
        self.shuffle_seed = self.method_config["shuffle_seed"]
        self.n_repreat = self.method_config["n_repreat"]
        self.multiprocessing = self.method_config["multiprocessing"]
        self.max_workers = self.method_config["max_workers"]
        self.debug = self.method_config["debug"]

        self.results_path = f"results/{self.dataset_name}/adas/{self.model_name_optimize}/{self.model_name_execute}"
        self.n_generation = self.method_config["n_generation"]
        self.debug_max = self.method_config["debug_max"]
        self.args = argparse.Namespace(**self.method_config)


    async def call_llm(self, prompt=None, system_prompt=None, messages=None):
        response = await super().call_llm(prompt=prompt, system_prompt=system_prompt, messages=messages,model_name=self.model_name_optimize)
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
        random.seed(self.shuffle_seed)

        examples = random.sample(val_dataset, len(val_dataset))

        if searching_mode:
            examples = examples[:self.valid_size] * self.n_repreat
        else:
            examples = examples[self.valid_size:self.valid_size + self.test_size] * self.n_repreat

        questions = [example['query'] for example in examples]
        answers = [example['solution'] for example in examples]

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

        for i,result in enumerate(results):
            if not isinstance(result,str):
                result = result.content
            if grade_answer(str(result), answers[i]):
                acc_list.append(1)
            else:
                acc_list.append(0)
        print(f"acc: {bootstrap_confidence_interval(acc_list)}")
        return acc_list

    def optimizing(self, val_dataset):
    # The original paper did not use the MATH dataset, here we use AFlow's MATH_val.
        optimized_path = Path(self.results_path) / "best_forward.txt"
        if optimized_path.exists():
            print(colored("The optimal forward function already exists!\n","red"))
            return
        best_fitness = 0


        # use validation dataset to optimize
        archive = get_init_archive()
        start = 0

        for solution in archive:
            if 'fitness' in solution:
                continue

            solution['generation'] = "initial"
            print(colored(f"============Initial Archive: {solution['name']}=================","yellow"))
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
            archive_path = os.path.join(self.results_path,"archive.json")
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            with open(archive_path, 'w') as json_file:
                json.dump(archive, json_file, indent=4)

        for n in range(start, self.n_generation):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            print(colored(f"============Generation {n + 1}=================","light_cyan"))
            system_prompt, prompt = get_prompt(archive)
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                ]
            
            try:
                msg_list = loop.run_until_complete(self.optimize(msg_list,archive,n))
            except Exception as e:
                print(f"During optimizing:\n{e}")
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
                        next_solution = asyncio.run(self.call_llm(msg_list))
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

            archive_path = os.path.join(self.results_path,"archive.json")
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            with open(archive_path, 'w') as json_file:
                json.dump(archive, json_file, indent=4)      

                
        
        forward_path = os.path.join(self.results_path,"best_forward.txt")
        with open(forward_path,"w") as f:
            f.write(best_solution['code'])
        print(colored("Optimization complete!","green"))
        print(colored(f"\n>> Optimization token stats: {self.get_token_stats()}","light_yellow"))
        token_path = os.path.join(self.results_path,"api_token.json")
        with open(token_path,"a") as f:
            json.dump(optimize_execute_token_stats, f, indent=4)

    async def optimize(self,msg_list,archive,n):
        next_solution = await self.call_llm(messages=msg_list)
        print(next_solution)
        Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)

        # Reflexion 1
        msg_list.append({"role": "assistant", "content": str(next_solution)})
        msg_list.append({"role": "user", "content": Reflexion_prompt_1})
        next_solution = asyncio.run(self.call_llm(messages=msg_list))

        # Reflexion 2
        msg_list.append({"role": "assistant", "content": str(next_solution)})
        msg_list.append({"role": "user", "content": Reflexion_prompt_2})
        next_solution = asyncio.run(self.call_llm(messages=msg_list))
        next_solution = next_solution.strip().replace('```json', '').replace('```', ',')
        next_solution = json.loads(next_solution)
        return







    def inference(self, query):
        global execute_model, api_key, base_url, SEARCHING_MODE
        SEARCHING_MODE = False
        execute_model = self.inference_model
        self.execute_model_dict = self.model_api_config[execute_model]['model_list'][0]
        api_key = self.execute_model_dict['api_key']
        base_url = self.execute_model_dict['model_url']
        self.max_workers = self.model_api_config[execute_model]["max_workers"]
        optimized_path = Path(self.results_path) / "best_forward.txt"
        if optimized_path.exists():
            with open(optimized_path,"r") as f:
                forward_str = f.read()
        else:
            raise NotImplementedError("Best_forward function does not exist!")

        namespace = {}
        exec(forward_str, globals(), namespace)
        names = list(namespace.keys())
        if len(names) != 1:
            raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
        func = namespace[names[0]]
        if not callable(func):
            raise AssertionError(f"{func} is not callable")
        setattr(AgentSystem, "forward", func)
        taskInfo = Info('task', 'User', query, -1)
        agentSystem = AgentSystem()
        response = agentSystem.forward(taskInfo)
        if not isinstance(response,str):
            response = response.content

        self.token_stats[self.model_name]["num_llm_calls"] = inference_execute_token_stats[self.model_name]["num_llm_calls"] 
        self.token_stats[self.model_name]["prompt_tokens"] = inference_execute_token_stats[self.model_name]["prompt_tokens"] 
        self.token_stats[self.model_name]["completion_tokens"] = inference_execute_token_stats[self.model_name]["completion_tokens"]
        return response      
    
        
        



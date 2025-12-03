import asyncio
import os
import random
import re
import shortuuid
import warnings

from copy import deepcopy
from collections import Counter
from pytube import YouTube
from termcolor import colored
from typing import Any,List,Optional,Dict

from .prompt import *

random.seed(0)
class BaseAgent():
    def __init__(self):
        self.id = shortuuid.ShortUUID().random(length=4)
        self.nodes = {}
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        
    def add_node(self, node):
        """
        Creates and adds a new node to the graph.
        If id is not provided, generates a unique id for the node.
        """
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=5)
        node.id = node_id
        self.nodes[node_id] = node
        return node 

class IO(BaseAgent):
    def __init__(self,env):
        super().__init__()
        io = DirectAnswer(env=env)
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]

class IO_MATH(BaseAgent):
    def __init__(self,env):
        super().__init__()
        io = DirectAnswer_MATH(env=env)
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]

class CodeReact(BaseAgent):
    def __init__(self,env,num_reacts: int = 1):
        self.num_reacts = num_reacts
        super().__init__()
        code_writing = CodeWriting(env=env)
        self.add_node(code_writing)
        last_node = code_writing
        for _ in range(self.num_reacts):
            code_rewrite = CodeWriting(env=env)
            last_node.add_successor(code_rewrite)
            last_node = code_rewrite
            self.add_node(code_rewrite)

        self.input_nodes = [code_writing]
        self.output_nodes = [code_rewrite]

    def run(self, inputs: Dict[str, Any], max_tries: int = 3, max_time: int = 600) -> List[Any]:
        def is_node_useful(node):
            if node in self.output_nodes:
                return True
            for successor in node.successors:
                if is_node_useful(successor):
                    return True
            return False
        
        useful_node_ids = [node_id for node_id, node in self.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(self.nodes[node_id].predecessors) for node_id in useful_node_ids}
        zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and node_id in useful_node_ids]

        for _ , input_node in enumerate(self.input_nodes):
            node_input = deepcopy(inputs)
            input_node.inputs = [node_input]

        while zero_in_degree_queue:
            current_node_id = zero_in_degree_queue.pop(0)
            current_node = self.nodes[current_node_id]
            tries = 0
            while tries < max_tries:
                try:
                    asyncio.run(self.nodes[current_node_id].execute())
                    break
                except asyncio.TimeoutError:
                    print(f"Node {current_node_id} execution timed out, retrying {tries + 1} out of {max_tries}...")
                except Exception as e:
                    print(f"Error during execution of node {current_node_id}: {e}")
                    break
                tries += 1

            for successor in current_node.successors:
                if successor.id in useful_node_ids:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

        final_answers = []

        for output_node in self.output_nodes:
            output_messages = output_node.outputs
            if len(output_messages) > 0:
                final_answer = output_messages[-1].get("output", output_messages[-1])
                final_answers.append(final_answer)
            else:
                for output_message in output_messages:
                    final_answer = output_message.get("output", output_message)
                    final_answers.append(final_answer)

        if len(final_answers) == 0:
            final_answers.append("No answer since there are no inputs provided")
        return final_answers
'''
class ToolTOT(BaseAgent):
    def __init__(self,env):
        super().__init__()
        query = GenerateQuery(self.domain, self.model_name)

        file_analysis = FileAnalyse(self.domain, self.model_name)
        web_search = WebSearch(self.domain, self.model_name)

        query.add_successor(file_analysis)
        query.add_successor(web_search)

        combine = CombineAnswer(self.domain, self.model_name)
        file_analysis.add_successor(combine)
        web_search.add_successor(combine)

        self.input_nodes = [query]
        self.output_nodes = [combine]

        self.add_node(query)
        self.add_node(file_analysis)
        self.add_node(web_search)
        self.add_node(combine)
'''

class Node():
    def __init__(self, operation_description: str, id: Optional[str], combine_inputs_as_one: bool,env,domenstrations=None):
        self.id = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.operation_description = operation_description
        self.predecessors: List[Node] = []
        self.successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.domenstrations = domenstrations if domenstrations else []
        self.combine_inputs_as_one = combine_inputs_as_one
        self.env = env

    def add_predecessor(self, operation: 'Node'):
        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: 'Node'):
        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self) 

    def process_input(self, inputs):
        all_inputs = []
        if inputs is None:
            if self.predecessors:

                for predecessor in self.predecessors:
                    predecessor_input = self.env.memory.get(predecessor.id, [])
                    if isinstance(predecessor_input, list) and predecessor_input:
                        predecessor_input = predecessor_input[-1]
                        all_inputs.append(predecessor_input)
                inputs = all_inputs
            else:
                raise ValueError("Input must be provided either directly or from predecessors.")
            
        elif not isinstance(inputs, list):
            inputs = [inputs]

        return inputs
    
    async def execute(self, **kwargs):
        self.outputs = []
        tasks = []
        # 1.Create tasks
        if not self.inputs and self.predecessors:
            if self.combine_inputs_as_one:
                combined_inputs = []
                for predecessor in self.predecessors:
                    predecessor_outputs = predecessor.outputs
                    if predecessor_outputs is not None and isinstance(predecessor_outputs, list):
                        combined_inputs.extend(predecessor_outputs)
                tasks.append(asyncio.create_task(self._execute(combined_inputs, **kwargs)))
            else:
                for predecessor in self.predecessors:
                    predecessor_outputs = predecessor.outputs
                    if isinstance(predecessor_outputs, list) and predecessor_outputs:
                        for predecessor_output in predecessor_outputs:
                            tasks.append(asyncio.create_task(self._execute(predecessor_output, **kwargs)))
        # There is direct input
        elif self.inputs:
            tasks = [asyncio.create_task(self._execute(input, **kwargs)) for input in self.inputs]
        else:
            warnings.warn("No input received.")
            return

        # 2.Perform tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if not isinstance(result, Exception):
                    if not isinstance(result, list):
                        result = [result]
                    self.outputs.extend(result)
                else:
                    print(colored(f"Node {type(self).__name__} failed to execute due to: {result.__class__.__name__}: {result}","light_red"))

class FinalDecision(Node):
    def __init__(self, operation_description: str = "Refer to all answers and give a final answer.",id=None,env=None):
        super().__init__(operation_description, id, True,env)

    async def _execute(self, inputs: List[Any] = [], **kwargs) -> None:
        prompt = None
        response = None
        if len(inputs) == 0:
            raise Exception("No inputs is not supported for MajorityVote")
        answers = [input.get("output") for input in inputs]
        counter = Counter(answers)
        sorted_counter = counter.most_common()
        max_freq = sorted_counter[0][1]
        equally_frequent_answers = [ans for ans, freq in sorted_counter if freq == max_freq]
        response = random.choice(equally_frequent_answers)
        # print(colored(f"{answers=} {response=}","blue"))

        executions = {
                        "task": inputs[0]["task"], 
                        "files": inputs[0]["files"],
                        "input": inputs, 
                        "subtask": prompt,
                        "output": response,
                        "format": "natural language"}

        if self.id not in self.env.memory:
            self.env.memory[self.id] = []
        self.env.memory[self.id].append(executions)

        return executions

class DirectAnswer(Node): 
    def __init__(self, operation_description: str = "Directly output an answer.",id=None,env=None):
        super().__init__(operation_description, id, True,env)
        
    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            task = input["task"]   
            messages = [{"role":"system", "content": MMLU_SYSTEM_PROMPT},
                       {"role":"user","content":task}]
            if self.env.inference_flag:
                response = self.env.call_llm(messages=messages)
            else:
                response = self.env.call_llm(messages=messages,model_name=self.env.model_name_execute)

            execution = {
                "task": task,
                "files": input.get("files", []),
                "input": task,
                "output": response,
                "ground_truth": input.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            if self.id not in self.env.memory:
                self.env.memory[self.id] = []
            self.env.memory[self.id].append(execution)

        return outputs 

class DirectAnswer_MATH(Node): 
    def __init__(self, operation_description: str = "Directly output an answer.",id=None,env=None):
        super().__init__(operation_description, id, True,env)
        
    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            task = input["task"]   
            messages = [{"role":"system", "content": MATH_SYSTEM_PROMPT},
                       {"role":"user","content":task}]
            if self.env.inference_flag:
                response = self.env.call_llm(messages=messages)
            else:
                response = self.env.call_llm(messages=messages,model_name=self.env.model_name_execute)

            execution = {
                "task": task,
                "files": input.get("files", []),
                "input": task,
                "output": response,
                "ground_truth": input.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            if self.id not in self.env.memory:
                self.env.memory[self.id] = []
            self.env.memory[self.id].append(execution)

        return outputs

class CodeWriting(Node):
    def __init__(
            self,
            env,
            operation_description: str = "a Python code generator",
            id=None,
            prompt=None,
            domenstrations=None
            ):
        super().__init__(operation_description, id, False,env,domenstrations)
        self.prompt = prompt if prompt else CODE_PROMPT
        self.max_domenstrations = 4

    def extract_example(self, prompt: str) -> list:
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    async def _execute(self, inputs: List[Any] = [], max_tries: int = 1, **kwargs):
        """
        Execute the node with the given inputs.
        """
        node_inputs = self.process_input(inputs)
        node_outputs = []

        for input in node_inputs:
            if input.get('is_solved', False):
                execution = deepcopy(input)
            else:
                task = input["task"]
                if 'feedback' in input.keys():
                    input = CODE_REACT_PROMPT.format(question=task, solution=input["output"], feedback=input["feedback"])
                else:
                    input = input["task"]
                self.internal_tests = self.extract_example(task)
                message = []
                message.append({"role":"system","content":self.prompt})
                for domenstration in self.domenstrations:
                    message.append({"role":"user","content":self.domenstration['input']})
                    message.append({"role":"assistant","content":self.domenstration['output']})
                message.append({"role":"user","content":input})
                if self.env.inference_flag:
                    response = self.env.call_llm(messages=message)
                else:
                    response = self.env.call_llm(messages=message,model_name=self.env.model_name_execute)
                response = response.strip("```python\n").strip("```")
                is_solved, feedback, _ = self.env.execute(response, self.internal_tests, timeout=10)
                execution = {
                    "task": task, 
                    "input": input,
                    "feedback": feedback,
                    "output": response,
                    "format": "python code",
                    "is_solved": is_solved,
                }
            if self.id not in self.env.memory:
                self.env.memory[self.id] = []
            self.env.memory[self.id].append(execution)
            node_outputs.append(execution)

        return node_outputs
    
    async def evaluate(self, candidate):
        prompt, domenstrations = candidate
        inputs = self.env.memory.get(self.id, [])
        inputs = [record for record in self.env.memory.get(self.id,[])[-10:]]
        score = 0
        for input in inputs:
            messages = []
            messages.append({"role":"system","content":prompt})
            for domenstration in domenstrations:
                messages.append({"role":"user","content":domenstration['input']})
                messages.append({"role":"assistant","content":domenstration['output']})
            messages.append({"role":"user","content":input['input']})
            response = self.env.call_llm(messages=messages)
            response = response.strip("```python\n").strip("```")
            tests = self.extract_example(input['task'])
            is_solved, _, _ = self.env.execute(response, tests, timeout=10)
            score += is_solved

        return score / len(inputs)
''' 
class GenerateQuery(Node):
    def __init__(self, 
                 env,
                 operation_description: str = "Given a question, return what infomation is needed to answer the question.",
                 id=None):
        super().__init__(operation_description, id, env,True)

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            # Regular expression for matching URLs
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, input["task"])
            download_paths = []

            # Process each URL
            for url in urls:
                if bool(re.match(youtube_regex, url)):
                    download_path = self._youtube_download(url)
                    if download_path:
                        download_paths.append(download_path)

            files = input.get("files", [])
            if not isinstance(files, list):
                files = []
            files.extend(download_paths)


            prompt =  GAIA_PROMPT.format(question=input["task"])           
            message = [{"role":"system", "content": GAIA_SYSTEM_PROMPT},
                       {"role":"user","content": prompt}]
            response = self.env.call_llm(messages=message)

            executions =  {
                           "task": input["task"], 
                           "files": files,
                           "input": input.get("task", None), 
                           "subtask": prompt,
                           "output": response,
                           "format": "natural language"}
            outputs.append(executions)
            if self.id not in self.env.memory:
                self.env.memory[self.id] = []
            self.env.memory[self.id].append(executions)
            
        return outputs

    def _youtube_download(self, url: str) -> str:
        try:
            video_id = url.split('v=')[-1].split('&')[0]
            video_id = video_id.strip()
            youtube = YouTube(url)
            video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not video_stream:
                raise ValueError("No suitable video stream found.")
            
            output_dir = "workspace_nopush/tmp"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{video_id}.mp4"
            video_stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
            return output_path
        
        except Exception as e:
            print(colored(f"Error downloading video from {url}: {e}","red"))
            return ""

class FileAnalyse(Node):
    def __init__(self, 
                 env,
                 operation_description: str = "Given a question, extract infomation from a file.",
                 id=None):
        super().__init__(operation_description, id, env,True)
        self.reader = GeneralReader()

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:
            query = input.get("output", "Please organize the information of this file.")
            files = input["files"]
            answer = ''
            for file in files:
                response = self.reader.read(query, file)
                if not (isinstance(self.reader.file_reader, IMGReader) or isinstance(self.reader.file_reader, VideoReader)):
                    prompt = self.prompt_set.get_file_analysis_prompt(query=query, file=response)
                    response = self.env.call_llm(prompt=prompt)
                answer += response + '\n'

            executions = {
                "operation": self.node_name,
                "task": input["task"], 
                "files": file,
                "input": query, 
                "subtask": f"Read the content of ###{file}, use query ###{query}",
                "output": response,
                "format": "natural language"
            }

            outputs.append(executions)
            self.memory.add(self.id, executions)
        return outputs
    
    def read(self, task, file):
        files_content = ""
        file_content = self.file_reader.read_file(file, task)
        suffix = file.split(".")[-1]

        if suffix in ['py', 'java', 'cpp', 'c', 'js', 'css', 'html', 'htm', 'xml']:
            files_content += f'\nThe {suffix} file contains:\n---\n{file_content[0]}'
            if file_content[1] != '':
                files_content += f'\nExecution result:\n{file_content[1]}'
            if file_content[2] != '':
                files_content += f'\nExecution error message:\n{file_content[2]}'
            files_content += '\n---'

        elif suffix in ['txt', 'jsonl', 'csv', 'json', 'jsonld', 'jsonl', 'yaml', 'yml', 
                        'xlsx', 'xls', 'jpg', 'png', 'jpeg', 'gif', 'bmp', 'mp3', 'wav', 
                        'ogg', 'mp4', 'avi', 'mkv', 'mov', 'pdf', 'doc', 'docx', 'ppt', 
                        'pptx', 'md', 'markdown', 'tex', 'zip', 'tar', 'gz', '7z', 'rar']:
            files_content += f'\nThe {suffix} file contains:\n---\n{file_content}\n---'

        return files_content


class WebSearch(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Given a question, search the web for infomation.",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
        self.searcher =self._get_searcher()
        
class CombineAnswer(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Combine multiple inputs into one.", 
                 max_token: int = 500,
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
'''
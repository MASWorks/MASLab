import concurrent
import sys
import re
import traceback
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed

from .op_prompt import *
from .sanitize import *
import asyncio

class Operator:
    def __init__(self, llm, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Custom(Operator):
    def __init__(self, llm, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        if self.llm.inference_flag:
            response = await self.llm.async_call_llm(prompt=prompt)
        else:
            response = await self.llm.async_call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
        return {"response":response}

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    def __init__(self, llm, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        try:
            if self.llm.inference_flag:
                response = await self.llm.async_call_llm(prompt=prompt)
            else:
                response = await self.llm.async_call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
            
            code_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>"
            match = re.search(code_pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                code = re.sub(r"^```(?:\w+)?\n?|```$", "", code, flags=re.MULTILINE).strip()
                if not code:
                    code = ""
                    
            response = {"code": code}

        except Exception as e:
            response =  {"error": str(e)}

        return response
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract code from markdown code blocks in the response.
        
        Args:
            text: The text containing possible markdown code blocks
            
        Returns:
            The extracted code as a string, or empty string if no code blocks found
        """
        # Look for Python code blocks (```python ... ```)
        python_pattern = r"```python\s*([\s\S]*?)\s*```"
        python_matches = re.findall(python_pattern, text)
        
        if python_matches:
            # Join all Python code blocks
            return "\n\n".join(python_matches)
        
        # If no Python blocks found, look for generic code blocks (``` ... ```)
        generic_pattern = r"```\s*([\s\S]*?)\s*```"
        generic_matches = re.findall(generic_pattern, text)
        
        if generic_matches:
            # Join all generic code blocks
            return "\n\n".join(generic_matches)
        
        # No code blocks found
        return ""

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback)
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        
        prompt = prompt + f"\n# Response format (must be strictly followed) (do not include any other formats except for the given XML format):\n<solution_letter>The letter of most consistent solution.</solution_letter>"   
    
        if self.llm.inference_flag:
            response = await self.llm.async_call_llm(prompt=prompt)
        else:
            response = await self.llm.async_call_llm(prompt=prompt,model_name=self.llm.model_name_execute)

        try:
            pattern = r"<(\w+)>(.*?)</\1>"
            matches = re.findall(pattern, response, re.DOTALL)
            found_fields = {match[0]: match[1].strip() for match in matches}
        except:
            pass
        if isinstance(found_fields, dict):
            response = found_fields
        else:
            response = {"response": response}


        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}
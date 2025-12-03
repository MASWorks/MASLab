import sys
import json
import traceback

from typing import List

from .op_prompt import *
from .sanitize import *

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
            response = self.llm.call_llm(prompt=prompt)
        else:
            response = self.llm.call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
        return {"response":response}

    
class CustomCodeGenerate(Operator):
    def __init__(self, llm, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        if self.llm.inference_flag:
            response = self.llm.call_llm(prompt=prompt)
        else:
            response = self.llm.call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
        extracted_code = sanitize(code=response, entrypoint=entry_point)
        return {"response":extracted_code}

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
        field_names=["thought","solution_letter"]
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")
        example_str = "\n".join(examples)
        prompt += f"""
### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.\n
{example_str}
"""     
        names=["thought","solution_letter"]
        types={"thought":str,"solution_letter":str}
        if self.llm.inference_flag:
            response = self.llm.call_llm(prompt=prompt)
        else:
            response = self.llm.call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
        response=self.llm.xml_extract(response,names,types)
        response={"solution_letter":response}
        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

class Test(Operator):
    def __init__(self, llm, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="MBPP")
                
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                if self.llm.inference_flag:
                    response = self.llm.call_llm(prompt=prompt)
                else:
                    response = self.llm.call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
                solution = sanitize(code=response, entrypoint=entry_point)
                
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                if self.llm.inference_flag:
                    response = self.llm.call_llm(prompt=prompt)
                else:
                    response = self.llm.call_llm(prompt=prompt,model_name=self.llm.model_name_execute)
                solution = sanitize(code=response, entrypoint=entry_point)
        
        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}

def extract_test_cases_from_jsonl(entry_point: str):
    file_path = "/MAS-LLM/datasets/data/aflow_mbpp_test.json"
    hardcoded_cases = {
        "remove_odd": "",
        "replace_spaces": "",
        "snake_to_camel": "",
        "Split": "",
        "swap_List": "",
        "square_Sum": "",
        "sort_sublists": "",
        "unique_sublists": "",
    }
    # Check if there are hardcoded test cases
    if entry_point in hardcoded_cases:
        return hardcoded_cases[entry_point]

    # If there are no hardcoded test cases, read from the file
    with open(file_path, "r") as file:
        data = json.load(file)  
        for item in data:
            if item.get("entry_point") == entry_point:
                return item.get("test")

    return None

def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function

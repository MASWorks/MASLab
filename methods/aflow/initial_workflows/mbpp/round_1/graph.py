from typing import Literal
from  ..template import operator 
from ..round_1 import prompt as prompt_custom 

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(self,name: str,env) -> None:
        self.name = name
        self.llm=env
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # await self.custom(input=, instruction="") 
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="") # But When you want to get standard code ,you should use customcodegenerator.
        return solution['response']

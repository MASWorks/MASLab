from typing import Literal
from  ..template import operator 
from ..round_1 import prompt as prompt_custom 

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]
class Workflow:
    def __init__(self,name: str,env) -> None:
        self.name = name
        self.llm=env
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response']
    
    
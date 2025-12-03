MMLU_SYSTEM_PROMPT = """
You are a knowlegable expert in question answering.
I will ask you a question.
I will also give you 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
You must choose the correct answer to the question.
Your response must be one of the 4 letters: A, B, C or D,corresponding to the correct answer.
Only one letter (A, B, C or D) is allowed in your answer.
"""

MATH_SYSTEM_PROMPT = """
You are a knowlegable expert in math.
I will ask you a math question.Please answer the question.
"""

CODE_PROMPT = """
You are an AI that only responds with only Python code.
You will be given a function signature and its docstring by the user.
Write your full implementation (restate the function signature).
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```
"""
CODE_REACT_PROMPT = """
Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:
{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""

META_PROMPT1 = """
Here is an example when a Python code generator gets wrong.
Input:
{input}
------------------
The output was:
{output}
------------------
It received the following feedback:
{feedback}
Identify a problem in a Python code generator from the given example and suggest how to prevent it without mentioning the specific example. 
Respond only one sentence.
"""

META_PROMPT2 = """
I'm trying to define a Python code generator by prompting.
My current prompt is:
"{prompt}"

To generate an improved prompt, consider the following:
{advice}
Generate an improved prompt within five sentences. Do not mention a specific task in the prompt!
The prompt should be wrapped with <START> and <END>.
"""

GAIA_SYSTEM_PROMPT = """
You are a a general AI assistant.
"""

GAIA_PROMPT = """
# Information Gathering for Question Resolution


Evaluate if additional information is needed to answer the question.
If a web search or file analysis is necessary, outline specific clues or details to be searched for.


## ❓ Target Question:
{question}


## 🔍 Clues for Investigation:
Identify critical clues and concepts within the question that are essential for finding the answer.
"""
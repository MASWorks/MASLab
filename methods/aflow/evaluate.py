import threading
import regex,re,json,inspect,time
from termcolor import colored
from typing import Any, List,Tuple,Callable,Dict,Optional
from math import isclose
from pathlib import Path
from pydantic_core import to_jsonable_python
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from .initial_workflows.math.template.sanitize import sanitize

def write_json_file(json_file: str, data: list, encoding: str = None, indent: int = 4):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=to_jsonable_python)

def extract_model_answer(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, text, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()

    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = re.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""


async def evaluate_math(problem: dict, graph: Callable, log_path:str) -> Tuple[str, str, str, int, float]:
    input_text = problem["query"]
    expected_output = problem["solution"]

    try:
        output = await graph(input_text)
        expected_answer = extract_model_answer(expected_output)
        predicted_answer = extract_model_answer(output)

        if math_equal(predicted_answer, expected_answer):
            uni_score, extracted_output =  1, predicted_answer
        else:
            uni_score, extracted_output =  0, predicted_answer
            
        if uni_score == 0:
            log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=get_function_code(extract_model_answer),
                    log_path=log_path
                )
        return input_text, output, expected_output, uni_score

    except Exception as e:
        print(colored(f"Maximum retries reached. Skipping this sample. Error: {e}","light_red"))
        return input_text, str(e), expected_output, 0.0
    
def math_equal(prediction: Any, reference: Any) -> bool:
    if str(prediction) == str(reference):
        return True
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            return isclose(prediction, reference, abs_tol=1e-3)
    except:
        pass

    try:
        return symbolic_equal(prediction, reference)
    except:
        pass
    return False

def is_digit(num):
    return parse_digits(num) is not None

def parse_digits(num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
# When the original input is a percentage in LaTeX format (e.g., 50\%), 
# a backslash remains after processing, causing the float conversion to 
# fail returning None, and subsequent math operations may produce type errors.
# num = num.replace("\\%", "").replace("%", "")
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

def get_function_code(func):
    try:
        source_code = inspect.getsource(func)
        return source_code
    except OSError:
        return "no code"
    
def symbolic_equal(a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False
def log_mismatch(problem: str,expected_output: Any,prediction: str,extracted_output: Any,extract_answer_code: str = "None",log_path=None):
    log_data = {
        "question":problem,
        "right_answer": expected_output,
        "model_output": prediction,
        "extracted_output": extracted_output,
        "extract_answer_code": extract_answer_code,
            }
        
    log_file = Path(log_path) / "log.json"
    if log_file.exists():
        with log_file.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(log_data)
    write_json_file(log_file, data, encoding="utf-8", indent=4)

async def evaluate_mbpp(data: dict, graph: Callable,log_path:str) -> Tuple[str, str, str, float, float]:
    input_text = data["prompt"]
    expected_output = "\nCorrect Solution:\ndef " + data["code"]

    try:
        # Generate prediction using the graph function
        prediction = await graph(input_text, data["entry_point"])
        # Check the solution
        ret = check_solution(prediction, data["test"], data["entry_point"])
        test_case_details = ret[1]
        expected_output = test_case_details + "\nCorrect Solution:" + data["code"]

        # Calculate score based on the check result
        score = 1.0 if ret[0] == "PASS" else 0.0

        # Log mismatch if the score is 0
        if score == 0:
            log_mismatch(input_text, expected_output, prediction, score,log_path=log_path)

        return input_text, prediction, expected_output, score

    except Exception as e:
        print(colored(f"Maximum retries reached. Skipping this sample. Error: {e}","light_red"))
        return input_text, str(e), expected_output, 0.0
    
def check_solution(solution, test, entry_point):
    solution = sanitize(code=solution, entrypoint=entry_point)
    try:
        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": __import__("re"),
            "List": List,
            "Dict": Dict,
            "Tuple": Tuple,
            "Optional": Optional,
            "Any": Any,
        }

        exec(solution, global_dict)

        if entry_point not in global_dict:
            raise ValueError(f"Function {entry_point} is not defined in the solution.")

        exec(test, global_dict)

        check = global_dict["check"]

        result = run_with_timeout(check, 15)

        if result is None:
            result = ("PASS", "The solution passed all test cases.")

    except Exception:
        result = (
            "FAIL",
            "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
        )
    except Exception as e:
        error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
        result = ("FAIL", error_message)

        with open("error.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

    return result

def run_with_timeout(func, timeout):
    result = []
    stop_event = threading.Event()

    def target():
        try:
            result.append(func())
        except Exception as e:
            result.append(e)
        finally:
            stop_event.set()

    thread = threading.Thread(target=target)
    thread.start()
    is_timeout = not stop_event.wait(timeout)

    if is_timeout:
        raise Exception("Function execution timed out")

    if not result:
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]
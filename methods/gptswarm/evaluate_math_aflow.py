import regex,re
from termcolor import colored
from typing import Any
from math import isclose
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

def extract_model_answer(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, text, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()

    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = re.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""

def grade_answer(output, expected_output):
    try:
        expected_answer = extract_model_answer(expected_output)
        predicted_answer = extract_model_answer(output)
        if math_equal(predicted_answer, expected_answer):
            uni_score = True
        else:
            uni_score = False
        return uni_score

    except Exception as e:
        print(colored(f"Maximum retries reached. Skipping this sample. Error: {e}","light_red"))
        return False
    
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


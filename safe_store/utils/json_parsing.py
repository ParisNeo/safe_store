import re
import json
from ascii_colors import ASCIIColors, trace_exception

import re
import json
from collections import OrderedDict

def robust_json_parser(json_string: str) -> dict:
    json_string = re.sub(r"^```(?:json)?\s*|\s*```$", '', json_string.strip())

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as ex:
        err = ex

    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', json_string)
    cleaned_string = json_match.group(0) if json_match else json_string

    try:
        cleaned_string = re.sub(r'\bTrue\b', 'true', cleaned_string)
        cleaned_string = re.sub(r'\bFalse\b', 'false', cleaned_string)
        cleaned_string = re.sub(r'\bNone\b', 'null', cleaned_string)
        cleaned_string = re.sub(r'\b(undefined|NaN|Infinity|-Infinity)\b', 'null', cleaned_string)

        cleaned_string = re.sub(r'//.*', '', cleaned_string)
        cleaned_string = re.sub(r'/\*[\s\S]*?\*/', '', cleaned_string)

        cleaned_string = re.sub(r'\\([_`*#\-])', r'\1', cleaned_string)
        cleaned_string = re.sub(r',\s*(\}|\])', r'\1', cleaned_string)

        cleaned_string = re.sub(r'\}\s*\{', '},{', cleaned_string)

        def escape_newlines_in_strings(text: str) -> str:
            in_string = False
            result = []
            i = 0
            while i < len(text):
                c = text[i]
                if c == '"' and (i == 0 or text[i - 1] != '\\'):
                    in_string = not in_string
                if in_string and c == '\n':
                    result.append('\\n')
                else:
                    result.append(c)
                i += 1
            return ''.join(result)

        cleaned_string = escape_newlines_in_strings(cleaned_string)

        def escape_unescaped_inner_quotes(text: str) -> str:
            def fix(match):
                s = match.group(0)
                inner = s[1:-1]
                inner_fixed = re.sub(r'(?<!\\)"', r'\\"', inner)
                return f'"{inner_fixed}"'
            return re.sub(r'"(?:[^"\\]|\\.)*"', fix, text)

        cleaned_string = escape_unescaped_inner_quotes(cleaned_string)

        cleaned_string = re.sub(
            r"(?<=[:\[,])\s*'([^']*?)'\s*(?=[,\}\]])", 
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"', 
            cleaned_string
        )
        cleaned_string = re.sub(
            r"(?<=\{)\s*'([^']*?)'\s*:", 
            lambda m: '"' + m.group(1).replace('"', '\\"') + '":', 
            cleaned_string
        )
        cleaned_string = re.sub(r'(?<={|,)\s*([A-Za-z0-9_]+)\s*:', r'"\1":', cleaned_string)

        cleaned_string = cleaned_string.replace("...", "null")

        cleaned_string = re.sub(r'[\x00-\x1F\x7F\u00A0]', '', cleaned_string)
        cleaned_string = cleaned_string.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        cleaned_string = re.sub(r'"\s*\n\s*"', '"\\n"', cleaned_string)

        def balance_brackets(s: str) -> str:
            stack = []
            for c in s:
                if c in "{[":
                    stack.append(c)
                elif c in "}]":
                    if stack and ((stack[-1] == '{' and c == '}') or (stack[-1] == '[' and c == ']')):
                        stack.pop()
            for opener in reversed(stack):
                s += '}' if opener == '{' else ']'
            return s

        cleaned_string = balance_brackets(cleaned_string)

        return json.loads(cleaned_string, object_pairs_hook=OrderedDict)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Final error: {e}") from e

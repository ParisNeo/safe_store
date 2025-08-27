import re
import json
from ascii_colors import ASCIIColors, trace_exception

def robust_json_parser(json_string: str) -> dict:
    """
    Parses a possibly malformed JSON string using a series of corrective strategies.

    Args:
        json_string: A string expected to represent a JSON object or array.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If parsing fails after all correction attempts.
    """

    # STEP 0: Remove code block wrappers if present (e.g., ```json ... ```)
    json_string = re.sub(r"^```(?:json)?\s*|\s*```$", '', json_string.strip())

    # STEP 1: Attempt to parse directly
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as ex:
        err = ex

    # STEP 2: Extract likely JSON substring
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', json_string)
    cleaned_string = json_match.group(0) if json_match else json_string

    try:
        # STEP 3a: Normalize Python/JS booleans/nulls
        cleaned_string = re.sub(r'\bTrue\b', 'true', cleaned_string)
        cleaned_string = re.sub(r'\bFalse\b', 'false', cleaned_string)
        cleaned_string = re.sub(r'\bNone\b', 'null', cleaned_string)

        # STEP 3b: Remove comments (single-line and block)
        cleaned_string = re.sub(r'//.*', '', cleaned_string)
        cleaned_string = re.sub(r'/\*[\s\S]*?\*/', '', cleaned_string)

        # STEP 3c: Remove bad escape sequences like \_ or \*
        cleaned_string = re.sub(r'\\([_`*#\-])', r'\1', cleaned_string)

        # STEP 3d: Remove trailing commas
        cleaned_string = re.sub(r',\s*(\}|\])', r'\1', cleaned_string)

        # STEP 3e: Escape unescaped newlines inside string literals
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

        # STEP 3f: Escape unescaped inner double quotes inside strings
        def escape_unescaped_inner_quotes(text: str) -> str:
            def fix(match):
                s = match.group(0)
                inner = s[1:-1]
                # Escape double quotes that aren't already escaped
                inner_fixed = re.sub(r'(?<!\\)"', r'\\"', inner)
                return f'"{inner_fixed}"'
            return re.sub(r'"(?:[^"\\]|\\.)*"', fix, text)

        cleaned_string = escape_unescaped_inner_quotes(cleaned_string)

        # STEP 3g: Convert single-quoted strings to double quotes (arrays or object keys)
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

        # STEP 3h: Remove non-breaking spaces and control characters
        cleaned_string = re.sub(r'[\x00-\x1F\x7F\u00A0]', '', cleaned_string)

        # STEP 3i: Fix smart quotes
        cleaned_string = cleaned_string.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # STEP 3j: Remove line breaks between JSON tokens that don't belong
        cleaned_string = re.sub(r'"\s*\n\s*"', '"\\n"', cleaned_string)

        # Final parse
        return json.loads(cleaned_string)

    except json.JSONDecodeError as e:
        ASCIIColors.magenta("\n--- JSONDecodeError ---")
        ASCIIColors.red(e)
        ASCIIColors.magenta("\n--- Original String ---")
        ASCIIColors.yellow(json_string)
        ASCIIColors.magenta("\n--- Final Cleaned String Attempted ---")
        ASCIIColors.red(cleaned_string)
        trace_exception(err)
        raise ValueError(f"Failed to parse JSON. Final error: {e}") from e

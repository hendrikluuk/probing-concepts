from typing import Any

def is_valid_response(response: Any) -> bool:
    return isinstance(response, list) or isinstance(response, dict) or is_valid_response_str(response)

def is_valid_response_str(x: Any) -> bool:
    if not isinstance(x, str):
        return False

    if '{' in x or '}' in x:
        # seems like the string is an invalid JSON, othwrwise it would have been cast as a Python object
        return False

    return x and 'error when' not in x.lower()
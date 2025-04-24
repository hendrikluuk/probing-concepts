"""
  Adjust the code in this file to match your API access pattern
"""
import re
import json
import time as timeit
from typing import Any

import requests

json_pattern = re.compile(".*```json([^`]+)```.*")

def fetch(params:Any, time:bool=False, verbose:bool=False, complete_response:bool=False, timeout:int=180) -> Any:
    """ Blocking HTTP request """

    if time:
        start = timeit.time()

    try:
        if "body" in params:
            response = requests.post(params["url"], json=params["body"], verify=False, timeout=timeout)
        else:
            response = requests.get(params["url"], verify=False, timeout=timeout)
        if not response.ok:
            print(f"HTTP_fetch: ok={response.ok} text='{response.text}' reason='{response.reason}'")

    except requests.exceptions.Timeout:
        print(f'Request to \'{params["url"]}\' timed out after {timeout} seconds')
        return

    if time:
        elapsed = timeit.time() - start
        print(f"completed in {elapsed:.3f} seconds")

    result = response.json()

    if complete_response:
        return result

    if verbose:
        print(f"Response: {result}")
    if "status" in result and result["status"] == "ok":
        if result.get("result"):
            try:
                response = result["result"]["response"]
                response = re.sub(json_pattern, "\\1", response).strip()
                try:
                    data = json.loads(response)
                    return data
                except:
                    if not response:
                        return result["result"]
                    return response
            except:
                return result["result"]

    print(f"Unexpected response detected!\n{response.text}")
    return result

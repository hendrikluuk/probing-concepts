import re
import json

from utils import concept_in_response, scoring_model, base_url, prompt_template_invocation_path, default_body, base_url
from utils.fetch import fetch

call = "concept-equivalence"
call_url = f"{base_url}{prompt_template_invocation_path}{call}"

def score_decide_concept(response:dict, concepts:list[dict]) -> dict:
    assert ("selection_criteria" in response["context"] or "definition" in response["context"]) and isinstance(response["response"], str), \
        "Incompatible response: context must include 'selection_criteria' or 'definition' and response must be a string."
    
    baseline = None
    for concept in concepts:
        if concept["concept"] == concept_in_response(response):
            baseline = concept
            break

    assert baseline, f"Baseline concept '{concept_in_response(response)}' not found"
    context = map_context(response, baseline)

    body = {
        **default_body,
        "context": context
    }

    score = {
        "test": response["call"],
        "responder": response["model"],
        "concept": baseline["concept"],
        "domain": baseline["domain"],
        "response": response["response"],
        "judgement": {}
    }

    try:
        api_response = fetch({"url": call_url, "body": body}, complete_response=True)
    except Exception as e:
        print(f"Error while fetching response: {e}")
        return score

    score["judgement"], used_model = extract_response(api_response)

    score["scorer"] = used_model or scoring_model
    calculate_metrics(score)
    print(f"Score: {score}")
    return score

def map_context(response:dict, baseline:dict) -> dict:
    """ Map values from {response} and {baseline} to {context} """
    return {
        "conceptA": response["response"],
        "conceptB": baseline["concept"],
    }

def extract_response(api_response:str|dict) -> tuple[dict, str]:
    assert isinstance(api_response, dict), "API response must be a dictionary"

    used_model = None
    json_pattern = re.compile(".*```json([^`]+)```.*")

    response = {"error": "unidentified error"}

    if not api_response:
        response = {"error": "empty response for API"}

    elif api_response["status"] == "ok" and api_response.get("result"):
        api_result = api_response.get("result")
        used_model = api_result.get("model")
        response = api_result.get("response", "")
        response = re.sub(json_pattern, "\\1", response).strip()

        # one more attempt to catch json object in case {json_pattern} did not do the job
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            response = response[start:end]

        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error while decoding response:\n{response}")
            response = {"error": f"JSON decode error: {e}"}
    else:
        response = {"error": api_response.get('reason', 'unknown reason')}

    return response, used_model

def calculate_metrics(score:dict):
    if "equivalent" in score["judgement"]:
        TP = int(score["judgement"].get("equivalent", 0))
        FP = abs(TP - 1)
        score["judgement"]["TP"] = TP
        score["judgement"]["FP"] = FP

#!/usr/bin/env python3
"""
  Use an LLM as judge approach to decide which referents mentioned in the response are
  found in the baseline referents list. Since the baseline referents list can include tens
  of thousands of referents, we retrieve for each referent in the response the top 3 most
  similar ones based on embeddings and ask the LLM to decide whether any of these are referenced
  in the response.
"""
import re
import json

from utils import scoring_model, base_url, prompt_template_invocation_path, default_body, concept_in_response
from utils.subclasses import get_children
from utils.embedder import Embedder
from utils.fetch import fetch

call = "score-referents"
call_url = f"{base_url}{prompt_template_invocation_path}{call}"

embedder = Embedder()
# this can take a while when loaded for the first time
embedder.build_index()

def score_referents(response:dict, concepts:list[dict]) -> dict:

    baseline = None
    for concept in concepts:
        if concept["concept"] == concept_in_response(response):
            baseline = concept
            break

    result = {
        "test": response["call"],
        "responder": response["model"],
        "concept": response["context"]["concept"],
        "domain": baseline["domain"],
        "response": response["response"],
        "scorer": scoring_model,
        "judgement": {"TP": 0, "FP": 0}
    }

    if not baseline:
        print(f"WARNING: Baseline concept '{concept_in_response(response)}' not found")
        # return an empty result which should be discarded
        return {}

    if not response["response"] or isinstance(response["response"], str):
        # if the response is empty or a string, we cannot score it
        print(f"WARNING: Response is empty or invalid (will assign 0 score):\n{response}")
        return result

    context = map_context(response, baseline) 
    body = {
        **default_body,
        "context": context,
    }

    api_response = fetch({"url": call_url, "body": body}, complete_response=True)
    judgement = extract_response(api_response)
    if isinstance(context["baseline"], str):
        judgement["true_referents"] = json.loads(context["baseline"])
    else:
        judgement["true_referents"] = context["baseline"]

    calculate_metrics(judgement, call=response["call"])
    result["judgement"] = judgement

    return result

def map_context(response:dict, baseline:dict) -> dict:
    """ Map values from {response} and {baseline} to {context} """
    if response["context"].get("true_referents"):
        # this is a response to the 'decide-referents' prompt template
        true_referents = json.loads(response["context"]["true_referents"])
    else:
        true_referents = get_referents(response["response"], baseline)

    context = {
        "baseline": true_referents,
        "student_response": response["response"],
    }
    return context

def extract_response(api_response:dict) -> dict:
    """
        Extract the response from the API response and return it as a dictionary.
        Also return the model used to generate the response.
    """
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

    return response

def calculate_metrics(judgement:dict, call:str):
    """
        Calculate true positive, false positive and false negative count
         based on the judgement
    """
    if "matches" in judgement and "mismatches" in judgement:
        # set(*) makes sure that no response is counted twice
        try:
            TP = len(set(judgement["matches"]))
            FP = len(set(judgement["mismatches"]))
        except TypeError as e:
            print(f"Error while calculating TP/FP: {e}\njudgement: {judgement}")
            TP = 0
            FP = 0

        FN = 0 

        if call == "decide-referents":
            FN = len(judgement["true_referents"]) - TP

        judgement["TP"] = TP
        judgement["FP"] = FP
        judgement["FN"] = FN

def get_referents(response:list[str|dict|list], baseline:dict, max_response_len:int=24, top_k:int=3) -> list[str]:
    """
        Get the referents from the baseline and return them as a list of strings.
        When the number of referents in baseline is greater than {max_referents},
        the referents are filtered to only include the top {top_k} most similar ones to
        each referent in the response based on embedding distance.
    """
    if not isinstance(response, list):
        print(f"WARNING: Response is not a list: {response}")
        response = [f"{key}: {value}" for key, value in response.items()]

    assert isinstance(response, list), f"Response should be a list, but got {type(response)}"
    baseline_referents = get_children(baseline["referents"])
    max_referents = max_response_len * top_k
    result = set()

    for ref in response[:max_response_len]:
        if type(ref) in [dict, list]:
            ref = json.dumps(ref)
        if not ref:
            continue

        # if the referent is a string, get the top {top_k} most similar ones from the baseline
        if len(baseline_referents) > max_referents:
            similar_referents = embedder.search(ref, baseline['concept'], n=top_k)
            print(f"Found similar referents to '{ref}': {similar_referents}")
            result.update(similar_referents)
        else:
            result.add(ref)

    return sorted(result)

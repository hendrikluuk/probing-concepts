#!/usr/bin/env python3
"""
    Score the estimates of semantic field size algorithmically (without using the LLM as a judge).
"""
import math
from numbers import Number

from utils.various import concept_in_response
from utils.subclasses import get_children

# orders of magnitude for the point estimate deviation for an invalid response
point_estimate_deviation_for_invalid_response = 3

def score_semantic_field_size(response:dict, concepts:list[dict]) -> dict:
    score = {
        "judgement": {
            "bounds correct": 0,
            "point estimate deviation": point_estimate_deviation_for_invalid_response,
        },
        "scorer": "algorithmic",
        "responder": response["model"],
        "concept": response["context"]["concept"],
        "domain": response["context"]["domain"],
        "test": response["call"],
        "response": response["response"]
    }

    if not response["response"] or isinstance(response["response"], str):
        return score

    baseline = None
    for concept in concepts:
        if concept["concept"] == concept_in_response(response):
            baseline = concept
            break

    context = map_context(response, baseline)
    score["judgement"].update(calculate_metrics(context))
    return score

def map_context(response:dict, baseline:dict) -> dict:
    """
    Combine relevant information from the {response} and the {baseline} into a {context} for scoring purposes
    """
    context = {"response": convert_to_number(response["response"])}

    referents = baseline["referents"]
    if isinstance(referents, dict):
        # flatten the tree structure
        # include all children irrespective of the term length
        referents = get_children(baseline["referents"], max_word_len=10000)
    n_referents = len(referents)

    context["baseline"] = {"lower bound": 0, "upper bound": float('inf'), "point estimate": n_referents}
    bounds = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e9, 1e12]
    for i, b in enumerate(bounds):
        if n_referents >= b:
            context["baseline"]["lower bound"] = bounds[i]
        if n_referents < b:
            context["baseline"]["upper bound"] = bounds[i]
            break
    return context

def order_of_magnitude(response:dict, n:int) -> int:
    if not isinstance(n, Number) or not isinstance(response.get("point estimate"), Number):
        print(f"WARNING: Not a number: response={response}, n={n}")
        return point_estimate_deviation_for_invalid_response
    ratio = response["point estimate"] / n
    try:
        return int(abs(math.log10(ratio)))
    except OverflowError:
        # infinity does not work well with int
        pass
    except ValueError:
        # 0 does not work well with log10
        pass
    return point_estimate_deviation_for_invalid_response

def convert_to_number(response:dict) -> dict:
    for key in response:
        if response[key] in ["unlimited", "infinity"]:
            response[key] = float('inf')
            continue
        try:
            response[key] = eval(response[key])
        except:
            if response[key] == "1e12 < R":
                response[key] = float('inf')
    return response    

def calculate_metrics(context:dict) -> dict:
    """
        Calculate metrics for the response that was returned by the LLM
        {context} is the original context including the response that is being scored
    """
    baseline = context["baseline"]
    response = context["response"]
    score = {}

    score["bounds correct"] = 0
    # each correct bound is worth 0.5 points
    if response.get("upper bound") == baseline["upper bound"]:
        score["bounds correct"] += 0.5
    if response.get("lower bound") == baseline["lower bound"]:
        score["bounds correct"] += 0.5

    score["baseline"] = baseline
    # get the order of magnitude of the ratio between the point estimate and number of referents
    score["point estimate deviation"] = order_of_magnitude(response, baseline["point estimate"])
    return score
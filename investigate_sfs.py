#!/usr/bin/env python3
"""
  Load scores for the 'semantic-field-size' task. Select 
  1. Gpt4o and Gpt4o-min models
  2. Claude 3 sonnet and Claude 3 haiku models
  for the selected models, display their scores on the 'semantic-field-size' task
  and keep all responses where the smaller model achieves a higher score than the larger model.
  Output the filtered responses into a pretty-printed json file. 
"""
import json
import numbers
from typing import Any

from scipy.stats import mannwhitneyu

from utils.loaders import load_scores, load_concepts
from summarize_scores import semantic_field_size_accuracy

relevant_models = [
    {
        "larger": "anthropic.claude-3-sonnet-v1:0",
        "smaller": "anthropic.claude-3-haiku-v1:0"
    },
    {
        "larger": "gpt-4o",
        "smaller": "gpt-4o-mini"
    },
    {
        "larger": "meta.llama3-70b-instruct-v1:0",
        "smaller": "meta.llama3-8b-instruct-v1:0"
    }
]

def convert_to_number(value:Any) -> float|None:
    """
    Convert a value to an integer. If the value is a string, try to convert it to an integer.
    If the value is a float, round it to the nearest integer.
    If the value is already an integer, return it as is.
    """
    if isinstance(value, int):
        return value

    result = None
    try:
        result = float(value)
    except ValueError:
        if isinstance(value, str) and value.lower() in ["unlimited"]:
            # return largest representable value
            return float("inf")
    return result

def main():
    concept_list = load_concepts() 
    rm_set = set([model for model_pair in relevant_models for model in model_pair.values()])

    scores = load_scores()
    filtered_responses = {}
    for score in scores:
        if score["test"] == "semantic-field-size":
            if score["responder"] in rm_set:
                if score["responder"] not in filtered_responses:
                    filtered_responses[score["responder"]] = []
                filtered_responses[score["responder"]].append(score)

    # check the number of responses for each model
    for key in filtered_responses:
        assert(len(filtered_responses[key]) == len(concept_list))

    for key in filtered_responses:
        filtered_responses[key] = sorted(filtered_responses[key], key=lambda x: x["concept"])
        # highlight concepts that are missing from filtered responses
        # but are present in the concept list
        missing_concepts = set([concept["concept"] for concept in concept_list]) - set([score["concept"] for score in filtered_responses[key]])
        if missing_concepts:
            print(f"WARNING: missing concepts in '{key}' responses: {missing_concepts}")
    
    result = {}
    # filter for responses where the smaller model has a higher score than the larger model
    for model_pair in relevant_models:
        larger_model = model_pair["larger"]
        smaller_model = model_pair["smaller"]

        result[get_model_pair_key(model_pair)] = {
            "summary": {
                "total": len(filtered_responses[larger_model]),
                "bounds correct": {
                    "larger model wins": 0,
                    "smaller model wins": 0,
                    "ties": 0
                },
                "point estimate deviation": {
                    "larger model wins": 0,
                    "smaller model wins": 0,
                    "ties": 0
                },
                "point estimate": {
                    "larger model provides larger estimate": 0,
                    "larger model provides smaller estimate": 0,
                    "estimates are tied": 0
                },
                "accuracy": {
                    "larger model wins": 0,
                    "smaller model wins": 0,
                    "ties": 0
                }
            },
            "responses": [],
            "point estimates": {
                larger_model: [],
                smaller_model: []
            }
        }

        for score_l, score_s in zip(filtered_responses[larger_model], filtered_responses[smaller_model]):
            # make sure we are comparing the same concept
            assert(score_l["concept"] == score_s["concept"])
            if not isinstance(score_s["response"], dict):
                print(f"WARNING: response is not a dict for model '{score_s['responder']}' on concept '{score_s['concept']}': {score_s['response']}")
                continue
            if not isinstance(score_l["response"], dict):
                print(f"WARNING: response is not a dict for model '{score_l['responder']}' on concept '{score_l['concept']}': {score_l['response']}")
                continue

            pe_larger = convert_to_number(score_l["response"]["point estimate"])
            pe_smaller = convert_to_number(score_s["response"]["point estimate"])
            if pe_larger is None or pe_smaller is None:
                print(f"WARNING: point estimate {pe_larger} from '{score_l['responder']}' or {pe_smaller} from '{score_s['responder']}' on concept '{score_l['concept']}' is not a number")
                continue

            # add the point estimates to the result
            result[get_model_pair_key(model_pair)]["point estimates"][larger_model].append(score_l["response"]["point estimate"])
            result[get_model_pair_key(model_pair)]["point estimates"][smaller_model].append(score_s["response"]["point estimate"])

            if score_l["judgement"]["bounds correct"] == score_s["judgement"]["bounds correct"]:
                result[get_model_pair_key(model_pair)]["summary"]["bounds correct"]["ties"] += 1
            elif score_l["judgement"]["bounds correct"] > score_s["judgement"]["bounds correct"]:
                result[get_model_pair_key(model_pair)]["summary"]["bounds correct"]["larger model wins"] += 1
            else:
                result[get_model_pair_key(model_pair)]["summary"]["bounds correct"]["smaller model wins"] += 1

            if score_l["judgement"]["point estimate deviation"] == score_s["judgement"]["point estimate deviation"]:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate deviation"]["ties"] += 1
            elif score_l["judgement"]["point estimate deviation"] < score_s["judgement"]["point estimate deviation"]:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate deviation"]["larger model wins"] += 1
            else:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate deviation"]["smaller model wins"] += 1

            if score_l["response"]["point estimate"] == score_s["response"]["point estimate"]:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["estimates are tied"] += 1
            elif score_l["response"]["point estimate"] > score_s["response"]["point estimate"]:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["larger model provides larger estimate"] += 1
            else:
                result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["larger model provides smaller estimate"] += 1

            accuracy_l = semantic_field_size_accuracy(score_l)
            accuracy_s = semantic_field_size_accuracy(score_s)

            if accuracy_l == accuracy_s:
                result[get_model_pair_key(model_pair)]["summary"]["accuracy"]["ties"] += 1
            elif accuracy_l > accuracy_s:
                result[get_model_pair_key(model_pair)]["summary"]["accuracy"]["larger model wins"] += 1
            else:
                result[get_model_pair_key(model_pair)]["summary"]["accuracy"]["smaller model wins"] += 1

            if accuracy_l < accuracy_s:
                result_item = {key: score_l[key] for key in ["concept", "domain"]}
                for model in [larger_model, smaller_model]:
                    score = score_l if model == larger_model else score_s
                    score["judgement"]["accuracy"] = semantic_field_size_accuracy(score)
                    result_item[model] = {} 
                    for key in ["response", "judgement"]:
                        result_item[model][key] = score[key]
                result[get_model_pair_key(model_pair)]["responses"].append(result_item)

        larger_model_scores = result[get_model_pair_key(model_pair)]["point estimates"][larger_model]
        smaller_model_scores = result[get_model_pair_key(model_pair)]["point estimates"][smaller_model]

        # calculate paired mann-whitney U test for the point estimates to establish whether
        # the larger model is biased towards larger estimates
        u_statistic, p_value = mannwhitneyu(larger_model_scores, smaller_model_scores, alternative="greater")
        result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["mann-whitney U statistic"] = u_statistic
        result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["p-value"] = p_value
        result[get_model_pair_key(model_pair)]["summary"]["point estimate"]["p-value is significant"] = "yes" if p_value < 0.05 else "no"

        print(f"Summary for '{get_model_pair_key(model_pair)}': {json.dumps(result[get_model_pair_key(model_pair)]['summary'], indent=4)}")

    # output the result as a pretty-printed json file
    outfile = "reports/semantic_field_size_investigation.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Wrote results to '{outfile}'")

def get_model_pair_key(model_pair:tuple) -> str:
    return f"{model_pair['larger']} vs {model_pair['smaller']}"

if __name__ == "__main__":
    main()

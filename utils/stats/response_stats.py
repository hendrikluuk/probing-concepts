import os
import json

import pandas as pd

from utils import concept_in_response
from utils.subclasses import get_children
from utils.validators import is_valid_response

def response_stats(responses: list[dict], concepts: list[dict], tests: list[str], models: list[dict], outfile:str=os.path.join("reports", "response_stats.xlsx")):
    """
    Gather statistics about responses to the tasks and output as 'response_stats.xlsx'.

    The statistics include:
    - Number of valid responses
    - Number of invalid responses
    - Number of missing responses
    - Models that did not respond
    """
    if os.path.exists(outfile):
        print(f"WARNING: {outfile} already exists. Will skip response stats.")
        return

    models = [model["id"] for model in models]
    result = []

    # record the number of concepts subjected to each test per domain
    tests_per_domain = {}

    for concept in concepts:
        for test in tests:
            # in decide-referents, the {parent_concept} corresponds to the root concept
            responses_for_concept_test = [response for response in responses if concept_in_response(response) == concept["concept"] and response["call"] == test]

            # decide-referents will not produce a result file for concepts with a small number of subconcepts
            # so we can ignore missing responses for such concepts
            if not responses_for_concept_test:
                print(f"Missing responses for concept '{concept['concept']}' and test '{test}'")
                continue

            domain = concept["domain"]
            if domain not in tests_per_domain:
                tests_per_domain[domain] = {}
            if test not in tests_per_domain[domain]:
                tests_per_domain[domain][test] = 0
            tests_per_domain[domain][test] += 1

            valid_responses = []
            invalid_responses = []
            record = {"concept": concept["concept"], "test": test, "valid_responses": 0, "invalid_responses": 0, "missing_responses": 0, "missing_models": []}
            for response in responses_for_concept_test:
                model = response["model"]
                sanity_check(response, concept, test)
                if not is_valid_response(response["response"]):
                    print(f"Invalid response for model '{model}' on concept '{concept['concept']}' and test '{test}' (empty response, unexpected type or invalid JSON)")
                    invalid_responses.append(model)
                else:
                    valid_responses.append(model)

            record["valid_responses"] = len(valid_responses)
            record["invalid_responses"] = len(invalid_responses)
            valid_responses = set(valid_responses)
            invalid_responses = set(invalid_responses)
            missing_responses = set(models).difference(valid_responses.union(invalid_responses))
            record["missing_responses"] = len(missing_responses) 
            record["missing_models"] = list(missing_responses)
            result.append(record)

    # output a xlsx sorted by descending number of missing responses
    df = pd.DataFrame(result)
    df = df.sort_values(by="missing_responses", ascending=False)

    # write the tests per domain to the first sheet
    tests_per_domain_df = pd.DataFrame(tests_per_domain)

    with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Response Stats', index=False)
        tests_per_domain_df.to_excel(writer, sheet_name='Concepts Tested per Domain', index=True)

    print(f"Response stats written to '{outfile}'")

def sanity_check(response:dict, concept:dict, test:str):
    """
    Perform a sanity check on the response to a task.
    """
    # check if the concept is in the response
    if concept_in_response(response) != concept["concept"]:
        print(f"WARNING: Sanity check failed: concept '{concept['concept']}' not found in response '{response['response']}'")
        return False

    # check if the test is in the response
    if response["call"] != test:
        print(f"WARNING: Sanity check failed: test '{test}' not found in response '{response['response']}'")
        return False

    if test == 'decide-referents':
        subconcept = response['context']['concept']
        if not subconcept in concept['referents']:
            print(f"WARNING: Sanity check failed: subconcept '{subconcept}' not found in referents of '{concept['concept']}'")
            return False
        true_referents_in_context = set(json.loads(response['context']['true_referents']))
        true_referents = set(get_children(concept['referents'][subconcept]))
        if not true_referents_in_context.issubset(true_referents):
            print(f"WARNING: Sanity check failed: children of subconcept '{subconcept}' ({len(true_referents)}) do not include true_referents ({len(true_referents_in_context)}) {true_referents_in_context} from the context of test '{test}' in '{concept['concept']}'")
            raise
            return False

    return True
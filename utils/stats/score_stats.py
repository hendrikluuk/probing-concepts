import os
import pandas as pd
from utils.validators import is_valid_response

def score_stats(scores:list[dict], concepts:list[dict], tests:list[str], models:list[dict], outfile:str=os.path.join("reports", "score_stats.xlsx")):
    """
    Provide the following metrics for each model as a table where model is the row index and metrics are the columns:

    - Number of unique concepts the model was evaluated on
    - Number of tests the model was evaluated on (a test is a concept + test pair)    
    - Number of tests passed (scorer produced a positive score in response judgement)
        + selection criterion: TP > 0 OR ("bounds correct" > 0 OR point estimate deviation < 3)
        + sanity check: n_tests_passed <= n_tests
    - Number of tests failed
        + selection criteria: TP == 0 AND ("bounds correct" == 0 AND point estimate deviation >= 3)
        + sanity check: n_tests_failed + n_tests_passed = n_tests
        
    - Number of valid responses
        + selection criterion: is_valid_response == True
        + sanity check: n_valid_responses >= n_tests_passed
    - Number of invalid responses
        + selection criterion: is_valid_response == False (empty response or invalid JSON)
        + sanity check: n_invalid_responses + n_valid_responses = n_tests

    Export the results as 'score_stats.xlsx'.
    """
    if os.path.exists(outfile):
        print(f"WARNING: {outfile} already exists. Will skip score stats.")
        return

    # Create a list of models
    models = [model["id"] for model in models]
    result = []

    # capture invalid responses and output as an additional worksheet
    invalid_responses = []

    for model in models:
        record = {"model": model, "n_concepts": 0, "n_tests": 0, "n_tests_passed": 0, "n_tests_failed": 0, "n_valid_responses": 0, "n_invalid_responses": 0}
        observed_concepts = set()

        for concept in concepts:
            for test in tests:
                # in decide-referents, the {parent_concept} corresponds to the root concept
                scores_for_model_concept_test = [score for score in scores if score["responder"] == model and score["concept"] == concept["concept"] and score["test"] == test]
                
                # if there are no scores for this model/concept/test combination, skip it
                if not scores_for_model_concept_test:
                    if test != "decide-referents":
                        # 'decide referents' test is skipped for concepts that lack a sufficient number of descendants
                        print(f"WARNING: missing scores for model '{model}' on concept '{concept['concept']}' and test '{test}'")
                    continue

                # increment the number of concepts and tests
                observed_concepts.add(concept["concept"])
                record["n_tests"] += len(scores_for_model_concept_test)

                for score in scores_for_model_concept_test:
                    if is_valid_response(score["response"]):
                        record["n_valid_responses"] += 1
                    else:
                        record["n_invalid_responses"] += 1
                        invalid_responses.append({
                            "model": model,
                            "concept": concept["concept"],
                            "test": test,
                            "response": score["response"]
                        })

                    # check if the test passed
                    success_criteria = [
                        score["judgement"].get("TP", 0) > 0,
                        score["judgement"].get("bounds correct", 0) > 0,
                        score.get("point estimate deviation", 3) < 3
                    ]
                    
                    if any(success_criteria):
                        record["n_tests_passed"] += 1
                    else:
                        record["n_tests_failed"] += 1

        # sanity checks
        if record["n_tests_passed"] > record["n_tests"]:
            print(f"Sanity check failed: n_tests_passed ({record['n_tests_passed']}) > n_tests ({record['n_tests']})")
        if record["n_tests_failed"] + record["n_tests_passed"] != record["n_tests"]:
            print(f"Sanity check failed: n_tests_failed ({record['n_tests_failed']}) + n_tests_passed ({record['n_tests_passed']}) != n_tests ({record['n_tests']})")
        if record["n_valid_responses"] < record["n_tests_passed"]:
            print(f"Sanity check failed: n_valid_responses ({record['n_valid_responses']}) < n_tests_passed ({record['n_tests_passed']})")
        if record["n_invalid_responses"] + record["n_valid_responses"] != record["n_tests"]:
            print(f"Sanity check failed: n_invalid_responses ({record['n_invalid_responses']}) + n_valid_responses ({record['n_valid_responses']}) != n_tests ({record['n_tests']})")

        record["n_concepts"] = len(observed_concepts)
        result.append(record)

    with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
        df = pd.DataFrame(result)
        df = df.sort_values(by="n_invalid_responses", ascending=False)
        df.to_excel(writer, sheet_name='Score stats', index=False)

        # output the invalid responses as a separate worksheet
        invalid_df = pd.DataFrame(invalid_responses)
        invalid_df = invalid_df.sort_values(by="model")
        invalid_df.to_excel(writer, index=False, sheet_name="Invalid responses")

    print(df.head())
    print(f"Score stats saved to '{outfile}'")
#!/usr/bin/env python3
"""
Score responses to tasks using LLM as a judge approach.

The scoring tasks have been designed so as to minimize the preference of the scoring model towards its own 
responses. This is done by formulating the scoring task as a significantly simplified or weaker version 
of the original task where the scorer has access to the correct response.

For example, when the original task asked the model to generate referents to a given concept, 
the scoring task asks to determine overlapping entities between the task's response and a baseline list 
obtained from a human-curated source.
"""
import os
import json

from utils.loaders import load_concepts, load_responses
from utils.scorers import *

score_folder = "scores"

def score_responses():
    concepts = load_concepts()
    responses = load_responses()

    for response_batch in responses:
        filename = response_batch['file']
        outfile = f"{score_folder}/{filename}"
        result = []

        if os.path.exists(outfile):
            print(f"Skipping responses from {filename} because they have already been scored.")
            continue

        # each model can have multiple responses per concept and test
        for responses in response_batch['responses'].values():
            for response in responses:
                score = {}
                if 'decide-concept' in response['call']:
                    score = score_decide_concept(response, concepts)
                elif 'semantic-field' in response['call']:
                    score = score_semantic_field_size(response, concepts)
                else:
                    score = score_referents(response, concepts)
                if score:
                    # include successfully scored results only
                    result.append(score)

        if result:
            with open(outfile, "w") as f:
                json.dump(result, f, indent=4)

if __name__ == "__main__":
    score_responses()
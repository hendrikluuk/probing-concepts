#!/usr/bin/env python3
"""
  Output stats about the concepts, responses and scores that were evaluated
"""
from utils.loaders import *
from utils.stats import *

# create 'reports' folder if it does not exist
if not os.path.exists("reports"):
    os.makedirs("reports")

def main():
    concepts = load_concepts()
    concept_stats(concepts)

    responses = load_responses(flatten=True)
    tests = load_tests()
    models = load_models()
    response_stats(responses, concepts, tests, models)

    scores = load_scores()
    score_stats(scores, concepts, tests, models)

if __name__ == "__main__":
    main()

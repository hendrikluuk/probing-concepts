#!/usr/bin/env python3
"""
    Evaluate all models on each concept in all tests.
    Gather responses from all models in each test into a separate file.

 Pseudocode:
    1. load models from the models.json file, skip models with disabled flag
    2. load tests from the tests.json file
    3. iterate concept files from 'concepts' folder one by one and for each concept:
        4. for each test in tests:
            5. for each model in models:
                6. evaluate the model on the concept in the test
                7. store the response
            8. output responses from all models to the given concept and test pair to the 'responses' folder in file <concept>__<test_name>.json as
            a dictionary with keys as model names and values as responses

Implementation:
import the LLM calling function 'main' from utils/call_autofx.py as 'call_llm'

Usage:

1. evaluate all models on all concepts in all tests while not overwriting existing responses
./test_models.py 

2. evaluate all models on all concepts in a single test while overwriting existing responses
./test_models.py --test decide-referents --overwrite 

3. evaluate all models on a single concept in a single test
./test_models.py --test decide-referents --overwrite --concept D-ribose --verbose
"""
import os
import json
import argparse

from utils.call_llm import call_llm
from utils.map_context import map_context
from utils.loaders import *
from utils.validators import is_valid_response

# test all concepts in all tests with all models
def test_all(overwrite: bool = False, verbose: bool = False, concept: str = None, test: str = None, model: str = None):
    """
    Evaluate all models on each concept in all tests.
    Gather responses from all models in each test into a separate file.
    Do not overwrite existing responses by default unless they contain an error.
    """
    # load models, tests and concepts
    models = load_models()
    tests = load_tests()
    concepts = load_concepts()

    if model:
        models = [m for m in models if m['id'] == model]
    if test:
        tests = {test: tests[test]}
    if concept:
        concepts = [c for c in concepts if c['concept'].lower() == concept.lower()]

    assert models, "No models found"
    assert tests, "No tests found"
    assert concepts, "No concepts found"

    if verbose:
        print(f"Concepts: {len(concepts)}")
        print(f"Tests: {list(tests.keys())}")
        print(f"Models: {[m['id'] for m in models]}")

    try:
        # iterate over all concepts
        for concept in concepts:
            # iterate over all tests
            for test in tests.items():
                call, context_map = test
                outfile = f'responses/{concept["concept"].replace(" ", "_").lower()}__{test[0]}.json'
                responses_updated = False

                if os.path.exists(outfile) and not overwrite:
                    print(f"Loading existing responses from '{outfile}'...")
                    responses = json.load(open(outfile))
                else:
                    responses = {}

                # all models will be evaluated on the same context
                context = map_context(call, context_map, concept)

                for model in models:
                    # in case the result file already exists, rerun the test only
                    # for the models that have missing or failed responses in the result file
                    if model['id'] in responses and all([is_valid_response(r['response']) for r in responses[model['id']]]):
                        # skip models that have already been evaluated on this test
                        continue
                    response = test_model(call, concept, model, context)
                    if response:
                        responses[model['id']] = response
                        responses_updated = True

                if responses_updated:
                    # output responses from all models to the given concept and test pair
                    with open(outfile, 'w') as f:
                        json.dump(responses, f, indent=4)
                    print(f"Responses for '{outfile}' have been updated.")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    except Exception as e:
        print(f"Error: {e}")
        raise

def test_model(call: str, concept: dict, model: dict, context:dict) -> list[dict]:
    # for book keeping purposes and evaluation
    for key in ['domain', 'concept']:
        context[key] = concept[key]
    if not context:
        print(f"Skipping test '{call}' for concept '{concept['concept']}' (no context available)")
        return []

    if "siblings" in context:
        # Note! the prompt template will only use values from 'concept' and 'entities'
        # the rest of the context will be used for book keeping to facilitate evaluation
        result = []
        sibling_counter = 0
        # store the topmost concept for book keeping (will not be used by the prompt template)
        context['parent_concept'] = context['concept']
        for sibling, referents in context["siblings"].items():
            # concept will be used as an argument to the prompt template
            context['concept'] = sibling
            context['true_referents'] = referents 
            response = call_llm(call, {**context}, model['id'])
            if response:
                result.append(response)
            sibling_counter += 1
            if sibling_counter > 3:
                break
        return result

    response = call_llm(call, context, model['id'])
    if response:
        return [response]
    else:
        return []

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate all models on each concept in all tests.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing response files (be careful!)')
    parser.add_argument('--verbose', action='store_true', help='Print more information')
    parser.add_argument('--concept', help='Test only a specific concept')
    parser.add_argument('--test', help='Test only a specific test')
    parser.add_argument('--model', help='Test only a specific model')
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        print(args)

    test_all(**vars(args)) 
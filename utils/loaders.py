import os
import json

import pandas as pd

from utils.various import normalize_model_name, concept_in_response

def load_models() -> list[dict]:
    models = json.load(open('models.json'))
    return [model for model in models if not model.get('disabled')]

def load_tests() -> list[str]:
    return json.load(open('tests.json'))

def load_concepts(folder:str="concepts", concepts:list[str]|None=None, verbose:bool=True) -> list[dict]:
    """
    Load concepts from the specified {folder}.
    If a list of {concepts} is provided, filter the concepts by name.
    """
    files = [concept for concept in os.listdir(folder) if concept.endswith('.json')]
    data = [json.load(open(os.path.join(folder, file))) for file in files]
    if concepts:
        # filter concepts by name
        data = [concept for concept in data if concept['concept'].lower() in [c.lower() for c in concepts]]
    if verbose:
        print(f"Loaded {len(data)} concepts from {len(files)} files in '{folder}'\n")
    return sorted(data, key=lambda x: x['concept'])

def load_responses(folder:str="responses", flatten:bool=False) -> list[dict]:
    files = [concept_test for concept_test in os.listdir(folder) if concept_test.endswith('.json')]
    data = [{"file": file, "responses": json.load(open(os.path.join(folder, file)))} for file in files]
    known_concepts = set([concept["concept"] for concept in load_concepts(verbose=False)])

    if not flatten:
        return data

    result = []
    # iterate over all files 
    for fdata in data:
        concept, test = fdata['file'].split('__')
        concept = concept.replace('_', ' ')
        for model, responses in fdata["responses"].items():
            # one model can have multiple responses per test and concept (in case it was probed multiple times)
            result.extend(responses)
            for response in responses:
                if not 'concept' in response.get('context', {}):
                    try:
                        response['context']['concept'] = concept
                    except:
                        print(f"No context in response for concept '{concept}', test '{test}', model '{model}' in file '{fdata['file']}'")

    covered_concepts = set([concept_in_response(response) for response in result])
    unknown_concepts = covered_concepts.difference(known_concepts)
    if unknown_concepts:
        print(f"WARNING: found {len(unknown_concepts)} unknown concepts in the responses: {unknown_concepts}")

    return sorted(result, key=lambda x: (x['context']['concept'], x['call']))

def load_scores(folder:str="scores") -> list[dict]:
    files = [concept_test for concept_test in os.listdir(folder) if concept_test.endswith('.json')]
    result = []
    known_concepts = set([concept["concept"] for concept in load_concepts(verbose=False)])

    models = load_models()
    models = set([model['id'] for model in models])

    covered_concepts = set()
    covered_tests = set()

    for file in files:
        data = json.load(open(os.path.join(folder, file)))
        fconcept = file.split('__')[0].replace('_', ' ')
        scored_models = set()

        for score in data:
            scored_models.add(score['responder'])
            # check if the concept aligns in file name and in the scores with valid responses

            # make sure the concept aligns in file name and in the scores
            sconcept = score['concept']
            if sconcept.lower() != fconcept.lower():
                if 'decide-referents' in file:
                    score['concept'] = fconcept
                    score['subconcept'] = sconcept
                else:
                    print(f"WARNING: concept mismatch in file '{file}' ('{fconcept}') on score: '{sconcept}'")

            result.append(score)
            covered_concepts.add(score['concept'])
            covered_tests.add(score['test'])

        # check if all models scored
        missing_models = models.difference(scored_models)
        if missing_models:
            print(f"WARNING: missing scores for models {missing_models} in file '{file}'")

    unknown_concepts = covered_concepts.difference(known_concepts)
    if unknown_concepts:
        print(f"WARNING: found {len(unknown_concepts)} unknown concepts in the scores: {unknown_concepts}")

    print(f"Loaded {len(result)} scores for {len(models)} models on {len(covered_concepts)} concepts and {len(covered_tests)} tests from {len(files)} score files\n")
    return sorted(result, key=lambda x: (x['concept'], x['test'], x['responder']))

def load_score_summary(file_path:str = os.path.join('reports', 'summarized_scores.xlsx')) -> pd.DataFrame:
    """
    Load scores from the 'Scores' worksheet of the provided file_path.
    """
    df = pd.read_excel(file_path, sheet_name='Scores')

    # Sort the models based on the order in the model_order list
    model_order = [model['id'] for model in load_models()]
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df = df.sort_values('model')
    return df

def load_external_rankings(file_path:str = 'external_rankings.csv') -> pd.DataFrame:
    """
    Load external benchmarks from the provided file_path.
    """
    df = pd.read_csv(file_path)
    # Sort the models based on the order in the model_order list
    model_order = [normalize_model_name(model['id']) for model in load_models()]
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    df = df.sort_values('model')
    df.index = df['model']
    df = df.drop(columns=['model'])
    # scale to match internal accuracy estimates
    df = df.div(100)
    return df
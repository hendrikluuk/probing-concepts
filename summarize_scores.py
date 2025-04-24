#!/usr/bin/env python3
"""
Summarize scores across models, tasks and domains.
"""
import os

import pandas as pd

from utils.loaders import load_external_rankings, load_models, load_scores, load_tests
from utils.various import normalize_model_name
from utils.stats import phyper

def summarize_scores(outfile:str=os.path.join("reports", "summarized_scores.xlsx")):
    """
    Load scores and summarize as follows.

    Create a pandas data frame of responses with the following columns:
    - model (responder)
    - domain
    - test
    - concept
    - TP (true positives)
    - FP (false positives)
    - FN* (false negatives)
    - recall* (TP / (TP + FN))
    - accuracy (TP / (TP + FP + FN))

    for 'semantic-field-size' task, the accuracy is calculated as a ratio of max score:
        point_estimate_score = 3 - min(point_estimate_deviation, 3)
        range_score = bounds_correct * 3
        max_score = point_estimate_score + range_score
    * relevant only for 'decide-referents' task (for other tasks, FN is always 0)
    """
    scores = load_scores()
    models = [model['id'] for model in load_models()]
    tests = load_tests()

    # Create a list to hold the summarized data
    summarized_data = []
    for score in scores:
        model = score["responder"]
        if model not in models:
            # skip models that are not listed
            continue

        domain = score["domain"]
        test = score["test"]
        concept = score["concept"]
        TP, FP, FN = get_tp_fp_fn(score)

        # Calculate recall and accuracy
        # For 'decide-referents' task, FN is relevant
        # For other task recall will be always 1 (uninformative)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = get_accuracy(score)

        # Append the summarized data
        summarized_data.append({
            "model": model,
            "domain": domain,
            "test": test,
            "concept": concept,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "recall": recall,
            "accuracy": accuracy
        })

    # Convert the summarized data to a pandas DataFrame
    df = pd.DataFrame(summarized_data)

    # use this filter to exclude the 'semantic-field-size' test from the
    # final ranking because it did not correlate well with other tests
    no_sfs = df['test'] != "semantic-field-size"

    # summarize results for each model by domain and store in a separate DataFrame
    summary_df = df[no_sfs].groupby(['model', 'domain']).agg({
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'recall': 'mean',
        'accuracy': 'mean'
    }).reset_index()
    summary_df.rename(columns={
        'TP': 'Total TP',
        'FP': 'Total FP',
        'FN': 'Total FN',
        'recall': 'Average Recall',
        'accuracy': 'Average Accuracy'
    }, inplace=True)

    # create a separate ranking of models in each domain based on accuracy
    rankings = {}
    model_scores_per_domain = {}    

    # get the average ranking for each model based on accuracy across all domains
    avg_rankings = summary_df.groupby('model').agg({
        'Average Accuracy': 'mean'
    }).reset_index()
    model_scores_per_domain['Average'] = avg_rankings['Average Accuracy'].reset_index(drop=True)
    avg_rankings = avg_rankings.sort_values(by='Average Accuracy', ascending=False)
    rankings["average"] = avg_rankings.reset_index(drop=True)

    for domain in ["biology", "chemistry", "medicine"]:
        domain_df = summary_df[summary_df['domain'] == domain]
        model_scores_per_domain[domain.title()] = domain_df['Average Accuracy'].reset_index(drop=True)
        domain_df = domain_df.sort_values(by='Average Accuracy', ascending=False)
        rankings[domain] = domain_df.reset_index(drop=True)


    # Rank concepts and test pairs for average accuracy and sort by accuracy
    for domain in ["biology", "chemistry", "medicine"]:
        domain_df = df[df['domain'] == domain]
        domain_df = domain_df.groupby(['concept', 'test']).agg({
            'accuracy': 'mean'
        }).reset_index()
        domain_df = domain_df.sort_values(by='accuracy', ascending=False)
        rankings[f"{domain} concepts"] = domain_df.reset_index(drop=True)

    # Rank tests for average accuracy and sort by accuracy across all domains
    test_df = df.groupby(['test']).agg({
        'accuracy': 'mean'
    }).reset_index()
    test_df = test_df.sort_values(by='accuracy', ascending=False)
    rankings["tests"] = test_df.reset_index(drop=True)

    # record the rank of each model for each test in a separate DataFrame
    model_scores_per_test = []

    # For each test, rank models for average accuracy and sort by accuracy
    for test in tests.keys():
        test_df = df[df['test'] == test]
        test_df = test_df.groupby(['model']).agg({
            'accuracy': 'mean'
        }).reset_index()
        model_scores_per_test.append(test_df['accuracy'])

        test_df = test_df.sort_values(by='accuracy', ascending=False)
        rankings[test[:20]] = test_df.reset_index(drop=True)

    models = [normalize_model_name(model) for model in df['model'].unique()]
    model_scores_per_test = pd.concat(model_scores_per_test, axis=1, keys=tests.keys())
    model_scores_per_test.index = models
    model_scores_per_domain = pd.concat(model_scores_per_domain, axis=1, keys=["Average"] + ["Biology", "Chemistry", "Medicine"])
    model_scores_per_domain.index = models

    external_rankings = load_external_rankings()
    # merge external_rankings with model_scores_per_domain on both indices
    # NB! inner join will only keep models that have been evaluated in all benchmarks
    external_internal_by_domain = model_scores_per_domain.merge(external_rankings, left_index=True, right_index=True, how='inner')
    external_internal_by_test = model_scores_per_test.merge(external_rankings, left_index=True, right_index=True, how='inner')

    # Save the DataFrame to an Excel file
    if not os.path.exists(outfile):
        with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
            model_scores_per_domain.to_excel(writer, sheet_name='Performance across domains', index=True)

            # highlight the models that performed significantly better than expected based on the pool of responses given by all models
            phyper_upper_tail = phyper(df[no_sfs])
            phyper_lower_tail = phyper(df[no_sfs], lower_tail=True).drop(columns=['correct', 'correct (%)'])
            phyper_merged = pd.merge(phyper_upper_tail, phyper_lower_tail, left_index=True, right_index=True)
            phyper_merged.to_excel(writer, sheet_name='Performance outliers')

            external_internal_by_domain.to_excel(writer, sheet_name='Internal vs external ranking', index=True)
            external_internal_by_domain.corr(method='kendall').to_excel(writer, sheet_name='Domain cor w external ranking', index=True)

            for domain in rankings:
                rankings[domain].to_excel(writer, sheet_name=f"{domain.capitalize()} ranking", index=False)

            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, sheet_name='Scores', index=False)

            for method in ["pearson", "kendall"]:
                model_scores_per_test.corr(method=method).to_excel(writer, sheet_name=f'Test correlations ({method})', index=True)

            external_internal_by_test.corr(method='kendall').to_excel(writer, sheet_name='Test cor w external ranking', index=True)
        print(f"Scores summarized and saved to '{outfile}'")
    else:
        print(f"File '{outfile}' already exists. Skipping saving.")

def get_tp_fp_fn(score:dict) -> tuple:
    """
    Get TP, FP, FN from the score.
    """
    TP = score["judgement"].get("TP", 0)
    FP = score["judgement"].get("FP", 0)
    FN = score["judgement"].get("FN", 0)
    return TP, FP, FN

def get_accuracy(score:dict) -> float:
    """
    Calculate accuracy for the given score.
    """
    if score["test"] == "semantic-field-size":
        return semantic_field_size_accuracy(score)

    TP = score["judgement"].get("TP", 0)
    FP = score["judgement"].get("FP", 0)
    FN = score["judgement"].get("FN", 0)
    if TP + FP + FN == 0:
        return 0.0

    accuracy = TP / (TP + FP + FN)
    return accuracy

def semantic_field_size_accuracy(score:dict, include_point_estimate:bool=True, include_bounds:bool=True) -> float:
    """
    Calculate accuracy for 'semantic-field-size' task.
    """
    weight = 3.0
    max_score = sum([include_point_estimate, include_bounds]) * weight

    point_estimate_deviation = score["judgement"]["point estimate deviation"]
    bounds_correct = score["judgement"]["bounds correct"]
    point_estimate_score = weight - min(point_estimate_deviation, weight)
    range_score = bounds_correct * weight

    score = point_estimate_score * int(include_point_estimate) + range_score * int(include_bounds)
    
    accuracy = score / max_score
    return accuracy

if __name__ == '__main__':
    summarize_scores()
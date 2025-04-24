"""
M is the total number of objects
n is total number of Type I objects (e.g. correct responses).
The random variate represents the number of Type I objects in N drawn without replacement from the total population.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
"""
import pandas as pd
from scipy.stats import hypergeom, false_discovery_control

RED = '\033[91m'
END = '\033[0m'

def phyper(df:pd.DataFrame, lower_tail:bool = False, alpha:float=0.05) -> pd.DataFrame:
    """
    Calculate the hypergeometric probability of getting the observed number of correct answers or more 
    """
    score_field = "accuracy"
    models = df["model"].unique()

    # number of tests that were taken
    M = len(df)
    # number of tests that were answered correctly
    n = int(df[score_field].sum())

    print(f"Number of models: {len(models)}\nTotal number of responses: {M} ({M/len(models):.2f} per model)\nNumber of correct responses (type I objects): {n}\nNumber of incorrect responses (type II objects): {M-n}\n")
    print(f"Hypergeometric probability of getting the observed number of correct answers or {'less' if lower_tail else 'more'}:")
    results = []
    for model in models:
        model_df = df[df["model"] == model]
        # number of responses given (tests taken) by the {model}
        N = len(model_df)
        # number of correct responses given by the {model}
        correct = int(model_df[score_field].sum())

        if lower_tail:
            # calculate the probability of getting the observed number of correct responses or less
            p = hypergeom.cdf(correct, M, n, N)
        else:
            # calculate the probability of getting the observed number of correct responses or more
            p = hypergeom.sf(correct, M, n, N)
        results.append({"correct": correct, "model": model, "p": p, "N": N})

    # Opting for the more conservative Benjamini-Yekuteli, because we are expecting performance on different tasks to be correlated
    fdr = false_discovery_control(list(map(lambda x: x["p"], results)), method="by")

    tail = "lower" if lower_tail else "upper"
    records = []
    for i, (result, q) in enumerate(sorted(zip(results, fdr), key=lambda x: x[0]['correct']/x[0]['N'], reverse=True)):
        # print p in red color if below 0.05
        color_tag = RED if q < alpha else ""
        correct_ratio = result['correct'] / result['N']
        record = {"model": result['model'], "correct": result['correct'], "correct (%)": correct_ratio*100, f"p ({tail})": result["p"], f"q ({tail})": q}
        records.append(record)
        print(f"{i+1}. {color_tag}{result['model']}: {result['correct']}/{result['N']} ({correct_ratio*100:.1f}%) correct responses (fdr={q:.2e}){END}")

    print("")
    # save results to a dataframe
    df = pd.DataFrame(records)
    df = df.sort_values(by="correct (%)", ascending=False)
    df = df.set_index("model")
    return df
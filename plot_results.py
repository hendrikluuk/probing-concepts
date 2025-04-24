#!/usr/bin/env python3
"""
  Read scores from the 'Scores' worksheet of the 'summarized_scores.xlsx' file
  and make a panel of radar/spider plots (one per model) to visualize the accuracy
  in 6 dimensions (with each test as a dimension) ranging from 0 to 1.
"""
import os
import time

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px

from utils.loaders import load_concepts, load_score_summary, load_models
from utils.subclasses import get_children
from utils.various import normalize_model_name

by_test_plot_path = "./plots/average_accuracy_by_test.pdf"
by_domain_plot_path = "./plots/average_accuracy_by_domain.pdf"
semantic_field_size_histo_path = "./plots/semantic_field_size_histogram.pdf"

def plot_radar():
    """
    Plot scores from the 'summarized_scores.xlsx' file as radar/spider plots.
    Each plot represents the accuracy of a model across different tests.
    Add a title to the plot with the model name.
    The plot is saved as a PDF file.
    """
    # Load the scores from the Excel file
    df = load_score_summary()
    models = df['model'].unique()

    # Define the number of variables (tests)
    num_vars = len(df['test'].unique())

    # Create a radar/spider plot for each model
    # Organize plots into e 3x5 grid
    num_cols = 5
    num_rows = (len(models) + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axs = plt.subplots(num_rows, num_cols, subplot_kw=dict(polar=True), figsize=(15, 10))
    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    for i, model in enumerate(models):
        # Filter the DataFrame for the current model and keep only 'accuracy' and 'test' columns
        model_df = df[df['model'] == model][['accuracy', 'test']]

        # Calculate the average accuracy for each test
        model_df = model_df.groupby('test').mean().reset_index()
        model_df = model_df.sort_values(by='test')
        model_df = model_df.reset_index(drop=True)
        # Check if the model has scores for all tests
        if len(model_df) != num_vars:
            print(f"WARNING: model '{model}' does not have scores for all tests. Skipping plot.")
            continue

        # Prepare the data for plotting
        categories = [test_to_acronym[test] for test in model_df['test'].tolist()]
        values = model_df['accuracy'].tolist()

        # Set up the radar/spider plot
        angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]

        axs[i].set_theta_offset(3.14159 / 2)
        axs[i].set_theta_direction(-1)

        # Set the y-ticks and labels (set labels at 0.2 and 0.8)
        y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
        axs[i].set_yticks(y_ticks, [0, None, 0.5, None, 1.0], alpha=0.5)
        axs[i].set_xticks(angles, categories, color='grey', size=8)
        axs[i].set_ylim(0, 1.0)
        axs[i].plot(angles+angles[:1], values+values[:1], linewidth=1, linestyle='solid', label=model)
        axs[i].fill(angles, values, alpha=0.4)

        # Add a title to the plot with the model name
        axs[i].set_title(f"{normalize_model_name(model)}", size=11, color='black', y=1.2)
    # Save the plot as a PDF file
    plt.savefig(by_test_plot_path, bbox_inches='tight')
    plt.close()

def plot_bar():
    # Load the scores from the Excel file
    df = load_score_summary()

    # average results by model and domain
    df = df.groupby(['model', 'domain'], observed=True).agg({
        'accuracy': 'mean'
    }).reset_index()

    # apply tim_model to the model names
    df['model'] = df['model'].apply(normalize_model_name)

    # Create a color palette for the domains
    colors = px.colors.qualitative.Plotly
    color_palette = [colors[2], colors[0], colors[1]]

    fig = px.bar(df, x="model", y="accuracy", color="domain",
                 color_discrete_sequence=color_palette,
                 labels={'model': 'Model', 'accuracy': 'Accuracy', 'domain': 'Domain'},
                 height=600)
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Accuracy",
        legend_title="Domain",
        title_font=dict(size=20),
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 2.5]),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="black")
    fig.update_layout(legend=dict(title_font=dict(size=12), font=dict(size=10)))
    # Save the plot as a PDF file
    fig.write_image(by_domain_plot_path, format="pdf", width=1200, height=600)

def plot_histogram():
    """
    Plot the histogram of semantic field sizes across all concepts.
    """
    concepts = load_concepts()
    semantic_field_size = [len(get_children(concept["referents"])) for concept in concepts]

    ylim = 155
    nbins = 200
    cut_interval = [20, 120]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.04)
    
    # Create histograms using px but extract the traces
    hist1 = px.histogram(x=semantic_field_size, nbins=nbins)
    hist2 = px.histogram(x=semantic_field_size, nbins=nbins)
    
    # Add the first trace from each histogram to the subplot
    fig.add_trace(hist1.data[0], row=1, col=1)
    fig.add_trace(hist2.data[0], row=2, col=1)
    
    # Update axes
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Semantic Field Size", row=2, col=1)
    
    fig.update_yaxes(range=[cut_interval[1], ylim], row=1, col=1)
    fig.update_yaxes(range=[0, cut_interval[0]], title_text="Count", row=2, col=1)
    
    # output as pdf
    fig.write_image(semantic_field_size_histo_path, format="pdf", width=1200, height=600)

def plotly_hack(plot_path:str):
    """
    Plotly hack to avoid MathJax loading message.
    """
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(plot_path, format="pdf")
    time.sleep(1)


test_to_acronym = {
    "semantic-field-size": "SFS",
    "decide-referents": "DR",
    "limited-list-referents": "LLR",
    "limited-list-referents-from-selection-criteria": "LLR-SC",
    "decide-concept": "DC",
    "decide-concept-from-selection-criteria": "DC-SC"
}

if __name__ == "__main__":
    hack_path = "plots/plotly_hack.pdf"
    plotly_hack(hack_path)
    # delete the hack file
    if os.path.exists(hack_path):
        os.remove(hack_path)
    plot_radar()
    plot_bar()
    plot_histogram()
    print("Plots saved to ./plots directory.")

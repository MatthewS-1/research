import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_embedding_metrics(inference_model_id: str, input_json_path: str, output_png_path: str):
    sns.set_theme(style="whitegrid")

    # Load data
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Build a DataFrame with columns for language, metric, value
    rows = []
    for lang, metrics in data.items():
        for metric, value in metrics.items():
            rows.append({"language": lang, "metric": metric, "value": value})

    df = pd.DataFrame(rows)

    # Create the barplot
    g = sns.catplot(
        data=df, kind="bar",
        x="language", y="value", hue="metric",
        height=3.5, aspect=1.5,
        palette="Set3"
    )
    g.set_axis_labels("Language", "Averaged Cosine Similarity")
    plt.tight_layout()
    plt.legend(fontsize=7)
    plt.title(f"{inference_model_id.split('/')[1]} quantized from 16B to 8B")

    # Remove the default legend
    g._legend.remove()

    # Add bar labels
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=4)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    g.savefig(output_png_path, dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot embedding metrics from JSON and save a PNG barplot.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier, e.g., Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_png", type=str, required=True, help="Path to save output PNG")

    args = parser.parse_args()

    plot_embedding_metrics(args.model_id, args.input_json, args.output_png)
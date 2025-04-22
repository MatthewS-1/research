import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

inference_model_id = "Qwen/Qwen2.5-72B-Instruct" # "meta-llama/Llama-3.3-70B-Instruct" # 
FILENAME = f"/gscratch/ark/myrsong/multilingual_assumptions/results/outputs/embedding_plots/overall_plots/{inference_model_id.split('/')[1]}_data.json"
data = json.load(open(FILENAME, 'r'))

# Build a DataFrame with columns for language, metric, value, and group.
rows = []
for lang, metrics in data.items():
    for metric, value in metrics.items():
        rows.append({"language": lang, "metric": metric, "value": value})
        
df = pd.DataFrame(rows)        

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="language", y="value", hue="metric",
    height=3.5, aspect=1.5,  # Increase the size of the plot
    # legend=False,
    palette="Set3",       # Use a nicer color palette
)
g.set_axis_labels("Language", "Averaged Cosine Similarity")
# g.legend.set_title("Metric")
# Adjust subplot margins to prevent the "left cannot be >= right" error.
# g.figure.subplots_adjust(top=1, right=1)
plt.tight_layout()
plt.legend(fontsize=7)
plt.title(f"{inference_model_id.split('/')[1]} quantized from 16B to 8B")
g._legend.remove()
# g._legend.set_bbox_to_anchor((0, 0))  # Bottom left corner
# g._legend.set_loc("lower left")  # Align to lower left
# Add bar labels to show the numeric values
# We'll loop through each Axes object in the FacetGrid (often just one).
for ax in g.axes.flat:
    # ax.containers is a list of the bar containers (one per 'hue' category).
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=4)  # Show values with 2 decimal places

g.savefig(f"/gscratch/ark/myrsong/multilingual_assumptions/results/outputs/embedding_plots/overall_plots/{inference_model_id.split('/')[1]}_data.png", dpi=400)
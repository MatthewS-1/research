import json
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import logging
from pathlib import Path

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_embeddings(attribute, embeddings, colors, label_map, out):
    reduced = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(10, 6))
    colors = np.array(colors)
    avg = []
    for c in set(colors):
        idx = np.where(colors == c)
        avg.append(np.mean(embeddings[idx], axis=0))
        m = np.mean(reduced[idx], axis=0)
        plt.scatter(m[0], m[1], color=cm.tab20(c / len(label_map)), alpha=0.7, s=200,
                    edgecolors='black', linewidths=1.5,
                    marker="^" if c % 2 == 0 else "v")
        plt.scatter(reduced[idx, 0], reduced[idx, 1], color=cm.tab20(c / len(label_map)), alpha=0.01)
    handles = [plt.Line2D([0], [0], marker="^" if i % 2 == 0 else "v",
                          color='w', markersize=10,
                          markerfacecolor=cm.tab20(i / len(label_map)))
               for i in range(len(label_map))]
    plt.legend(handles, label_map.keys(), title="Lang-Prompt-Attr",
               loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title(f"Avg Embedding: {attribute}")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return avg


def plot_matrix(mat, labels, out, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(mat)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=6)
    plt.title(title)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(vals, labs, out):
    plt.figure(figsize=(10, 6))
    cols = ['skyblue' if i % 2 == 0 else 'lightcoral' for i in range(len(vals))]
    bars = plt.bar(labs, vals, color=cols)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-model-id", required=True)
    parser.add_argument("--embedding-model-id",
                        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    parser.add_argument("--use-leace", action="store_true")
    parser.add_argument("--input-dir", default="outputs/jsons")
    args = parser.parse_args()

    logging.set_verbosity_info()

    inp = Path(args.input_dir)
    data_file = inp / f"{args.inference_model_id.split('/')[-1]}_{'untranslated' if args.use_leace else 'translated'}.json"
    results = load_results(data_file)

    model = SentenceTransformer(args.embedding_model_id)

    spec = Path("outputs/graphs/specific_plots")
    ovrl = Path("outputs/graphs/overall_plots")
    jsn = Path("outputs/jsons")

    first = next(iter(results))
    attrs = results[first][next(iter(results[first]))].keys()
    overall = np.zeros((len(results)*2, len(results)*2))

    for attr in attrs:
        embs, labs, cols = [], [], []
        lmap, idx = {}, 0
        for lang, pm in results.items():
            for ptype, d in pm.items():
                lbl = f"{lang}-{ptype}-{attr}"
                if lbl not in lmap:
                    lmap[lbl] = idx; idx += 1
                labs += [lbl]*len(d[attr]); cols += [lmap[lbl]]*len(d[attr])
                embs += model.encode(d[attr], batch_size=100, show_progress_bar=True)

        avg = plot_embeddings(attr, embs, cols, lmap,
                              spec/f"{args.inference_model_id.split('/')[-1]}_{'untranslated_leace_' if args.use_leace else 'translated_'}scatter_{attr}.png")
        sim = cosine_similarity(np.array(avg))
        plot_matrix(sim, list(lmap.keys()),
                    spec/f"{args.inference_model_id.split('/')[-1]}_{'untranslated_leace_' if args.use_leace else 'translated_'}sim_{attr}.png",
                    f"Similarity: {attr}")
        overall += sim

    dp = jsn / f"{args.inference_model_id.split('/')[-1]}_data.json"
    data = {}
    if dp.exists():
        try:
            data = json.load(dp.open())
        except json.JSONDecodeError:
            pass

    overall /= len(attrs)
    fin_vals, fin_labs = [], []
    # skip the first language by starting at the 3rd index
    warnings.warn("Assuming first language is the default, check final data/plots to ensure this")
    for i, lang in zip(range(3, len(overall), 2), list(results.keys())[1:]):
        sub_label = "leace applied" if args.use_leace else "translated"
        fin_vals.extend( [overall[i,1], overall[i,i-1], overall[i-1, 1]] )
        fin_labs.extend( [f"language ({sub_label}) vs default", f"language ({sub_label}) vs location", f"location vs default"] )
        data.setdefault(lang, {})
        for val, lab in zip(fin_vals, fin_labs):
            data[lang][lab] = val
        

    plot_bar(fin_vals, fin_labs, ovrl/f"{args.inference_model_id.split('/')[-1]}_{'untranslated_leace_' if args.use_leace else 'translated_'}overall.png")

    dp.write_text(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()

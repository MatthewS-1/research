import os
import json
import pickle
import warnings
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_avg_shuffled_cosine(d_emb: np.ndarray, r_emb: np.ndarray, num_shuffles: int = 100) -> float:
    """
    Compute the average cosine similarity between two sets of embeddings
    by randomly shuffling and splitting them in half multiple times.
    """
    sims = []
    N = len(d_emb)
    half = N // 2

    for _ in range(num_shuffles):
        perm = np.random.permutation(N)
        first_half = d_emb[perm][:half]
        second_half = r_emb[perm][half:]

        mean_first = np.mean(first_half, axis=0).reshape(1, -1)
        mean_second = np.mean(second_half, axis=0).reshape(1, -1)

        sims.append(cosine_similarity(mean_first, mean_second)[0, 0])

    return float(np.mean(sims))


def run(inf_model_id: str, emb_model_id: str, eraser_path: str):
    warnings.warn("Assuming first language is the default")
    short = inf_model_id.rsplit("/", 1)[-1]
    untranslated_fp = f"outputs/jsons/{short}_untranslated.json"
    data_fp        = f"outputs/jsons/{short}_data.json"
    graphs_dir     = "outputs/graphs/specific_plots"
    os.makedirs(graphs_dir, exist_ok=True)

    default = json.load(open(untranslated_fp))
    data    = json.load(open(data_fp)) if os.path.exists(data_fp) else {}

    embedder = SentenceTransformer(emb_model_id)
    eraser   = pickle.load(open(eraser_path, "rb"))

    first_lang = next(iter(default))
    attrs      = default[first_lang]["language"].keys()
    langs      = list(default)

    def plot_and_save(vals, attr):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            vals.keys(),
            vals.values(),
            color=[("skyblue" if i % 2 == 0 else "lightcoral") for i in range(len(vals))]
        )
        for b in bars:
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/{short}_{attr}_baseline.png", dpi=300, bbox_inches="tight")
        plt.close()

    def run_exp(results: list[dict], names: list[str], use_leace: bool):
        all_attr_vals = defaultdict(list)
        for attr in tqdm(attrs, desc="Attributes"):
            vals = {}
            for res, nm in zip(results, names):
                d_emb = embedder.encode(default[first_lang]["language"][attr])
                r_emb = embedder.encode(res[first_lang]["language"][attr])

                if use_leace:
                    d_emb = eraser(torch.tensor(d_emb).cuda()).cpu().numpy()
                    r_emb = eraser(torch.tensor(r_emb).cuda()).cpu().numpy()

                avg = compute_avg_shuffled_cosine(d_emb, r_emb, 100)
                vals[f"{attr}-{nm}"] = avg
                all_attr_vals[nm].append(avg)

            plot_and_save(vals, attr)

        for nm, vals in all_attr_vals.items():
            data.setdefault(nm, {})
            data[nm][short] = sum(vals) / len(vals)

        with open(data_fp, "w") as f:
            json.dump(data, f, indent=4)

    # Experiment configurations
    exps = [
        # default vs default
        ([default], ["default vs default"], False),
        # default vs each translation + LEACE
        (
            [
                json.load(open(f"outputs/jsons/{short}_translated_to_{l}.json"))
                for l in langs[1:]
            ],
            [f"default vs default translated to {l} and used leace" for l in langs[1:]],
            True
        ),
        # default vs translation→back
        (
            [
                json.load(open(f"outputs/jsons/{short}_translated_to_{l}_and_back.json"))
                for l in langs[1:]
            ],
            [f"default vs default translated to {l} and back" for l in langs[1:]],
            False
        ),
    ]

    for results, names, leace in exps:
        run_exp(results, names, leace)


def main():
    parser = argparse.ArgumentParser(
        description="Run translation robustness experiments with/without LEACE."
    )
    parser.add_argument(
        "--inf-model-id",
        type=str,
        required=True,
        help="Inference model ID (used to locate JSON outputs)."
    )
    parser.add_argument(
        "--emb-model-id",
        type=str,
        required=True,
        help="SentenceTransformer model ID for computing embeddings."
    )
    parser.add_argument(
        "--eraser-path",
        type=str,
        required=True,
        help="Path to the pickled LEACE eraser model."
    )
    args = parser.parse_args()
    run(args.inf_model_id, args.emb_model_id, args.eraser_path)


if __name__ == "__main__":
    main()

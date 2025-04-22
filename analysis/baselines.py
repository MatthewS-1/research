import os, json, pickle, warnings
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def run(inf_model_id: str, emb_model_id: str, eraser_path: str):
    warnings.warn("Assuming first language is the default")
    # derive a short ID and file paths
    short = inf_model_id.rsplit("/", 1)[-1]
    untranslated_fp = f"outputs/jsons/{short}_untranslated.json"
    data_fp        = f"outputs/jsons/{short}_data.json"
    graphs_dir     = "outputs/graphs/specific_plots"
    os.makedirs(graphs_dir, exist_ok=True)

    # load default JSON + any existing data
    default = json.load(open(untranslated_fp))
    data    = json.load(open(data_fp)) if os.path.exists(data_fp) else {}

    # prep models
    embedder = SentenceTransformer(emb_model_id)
    eraser   = pickle.load(open(eraser_path, "rb"))

    # get the one “first” language key and its attributes
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
            plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/{short}_{attr}_baseline.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    def run_exp(results: list[dict], names: list[str], use_leace: bool):
        all_attr_vals = defaultdict(list)
        for attr in tqdm(attrs, desc="Attributes"):
            vals = {}
            for res, nm in zip(results, names):
                d_emb = embedder.encode(default[first_lang]["language"][attr])
                r_emb = embedder.encode(res[first_lang]["language"][attr])

                if use_leace:
                    # apply eraser on GPU, then back to CPU → np.array
                    d_emb = eraser(torch.tensor(d_emb).cuda()).cpu().numpy()
                    r_emb = eraser(torch.tensor(r_emb).cuda()).cpu().numpy()

                # 100 random‐shuffle cosine‐means
                sims = [
                    cosine_similarity(
                        np.mean(d_emb[p:=np.random.permutation(len(d_emb))][: len(d_emb)//2], axis=0).reshape(1, -1),
                        np.mean(r_emb[p][ len(d_emb)//2 :],           axis=0).reshape(1, -1)
                    )[0, 0]
                    for _ in range(100)
                ]
                avg = sum(sims) / 100
                vals[f"{attr}-{nm}"] = avg
                all_attr_vals[nm].append(avg)

            # make & save the bar chart for this attribute
            plot_and_save(vals, attr)

        # update JSON data
        for nm, vals in all_attr_vals.items():
            data.setdefault(nm, {})
            data[nm][short] = sum(vals) / len(vals)

        with open(data_fp, "w") as f:
            json.dump(data, f, indent=4)

    # build experiment configurations
    exps = [
        # default vs default
        ([default], [f"default vs default"],       False),
        # default vs each translation + LEaCE
        (
            [json.load(open(f"/gscratch/ark/myrsong/multilingual_assumptions/results/outputs/{short}_translated_to_{l}.json"))
             for l in langs[1:]],
            [f"default vs default translated to {l} and used leace" for l in langs[1:]],
            True
        ),
        # default vs translation→back
        (
            [json.load(open(f"/gscratch/ark/myrsong/multilingual_assumptions/results/outputs/{short}_translated_to_{l}_and_back.json"))
             for l in langs[1:]],
            [f"default vs default translated to {l} and back" for l in langs[1:]],
            False
        ),
    ]

    # run them all
    for results, names, leace in exps:
        run_exp(results, names, leace)

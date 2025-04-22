import argparse
import os
import pickle
import torch
from torch import Tensor
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from concept_erasure import LeaceEraser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate LEACE erasure on Flores101 data"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="eng,deu,fra,ita,por,hin,spa,tha",
        help="Comma-separated list of language codes"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        dest="embedding_model",
        help="SentenceTransformer model identifier"
    )
    parser.add_argument(
        "--eraser-path",
        type=str,
        default="multilingual_assumptions/results/analysis/eraser.pkl",
        dest="eraser_path",
        help="Path to load/save the LEACE eraser pickle"
    )
    return parser.parse_args()


def load_embeddings(langs: list[str], model: SentenceTransformer, split: str) -> tuple[Tensor, Tensor]:
    """
    Load embeddings and integer language labels for a specific dataset split.
    """
    X_parts, Y_parts = [], []
    for idx, lang in enumerate(langs):
        ds = load_dataset("gsarti/flores_101", lang, split=split)
        sentences = ds.get("sentence", [])
        if not sentences:
            raise ValueError(f"No sentences found in {lang} [{split}]")
        embs = model.encode(sentences, convert_to_tensor=True)
        X_parts.append(embs)
        Y_parts.append(torch.full((len(embs),), idx, dtype=torch.long))
    X = torch.vstack(X_parts)
    Y = torch.cat(Y_parts)
    return X, Y


def main():
    args = parse_args()
    languages = args.languages.split(",")
    num_langs = len(languages)

    # Prepare directories
    os.makedirs(os.path.dirname(args.eraser_path), exist_ok=True)
    embedding_model = SentenceTransformer(args.embedding_model)
    results_dir = os.path.join(
        os.path.dirname(args.eraser_path), "../outputs/embedding_plots"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Train logistic on dev before erasure
    X_dev, Y_dev = load_embeddings(languages, embedding_model, "dev")
    X_dev_cpu, Y_dev_cpu = X_dev.cpu(), Y_dev.cpu()
    clf = LogisticRegression(max_iter=1000).fit(X_dev_cpu, Y_dev_cpu)
    acc_before = accuracy_score(Y_dev_cpu, clf.predict(X_dev_cpu))
    print(f"Dev accuracy before erasure : {acc_before:.4f}")

    # Load or fit eraser
    if os.path.exists(args.eraser_path):
        with open(args.eraser_path, "rb") as f:
            eraser = pickle.load(f)
        print("Loaded existing eraser")
    else:
        one_hot = torch.nn.functional.one_hot(Y_dev, num_langs)
        eraser = LeaceEraser.fit(X_dev, one_hot)
        with open(args.eraser_path, "wb") as f:
            pickle.dump(eraser, f)
        print("Fitted and saved new eraser")

    # Evaluate on erased dev
    X_dev_erased = eraser(X_dev).cpu()
    clf_erased = LogisticRegression(max_iter=1000).fit(X_dev_erased, Y_dev_cpu)
    acc_after = accuracy_score(Y_dev_cpu, clf_erased.predict(X_dev_erased))
    print(f"Dev accuracy after erasure: {acc_after:.4f}")

    # Evaluate on devtest
    X_test, Y_test = load_embeddings(languages, embedding_model, "devtest")
    X_test_erased = eraser(X_test).cpu()
    acc_test = accuracy_score(Y_test.cpu(), clf.predict(X_test_erased))
    print(f"DevTest accuracy (after erasure): {acc_test:.4f}")

    # Plot results
    labels = [
        "Dev (before)",
        "Dev (after)",
        "DevTest (after)",
        "Random Guess"
    ]
    accuracies = [
        acc_before,
        acc_after,
        acc_test,
        1.0 / num_langs
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors[:len(labels)]
    bars = ax.bar(labels, accuracies, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("LEACE Erasure Accuracies on Flores101")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(results_dir, "leace.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

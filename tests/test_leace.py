import argparse
import os
import pickle
import sys
import torch
import pytest
import experiments.leace

# --- Dummy classes and fixtures for mocking external dependencies ---

class DummyDataset:
    def __init__(self, sentences):
        self._sentences = sentences
    def get(self, key, default=None):
        if key == "sentence":
            return self._sentences
        return default

class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, sentences, convert_to_tensor=False):
        # Return a tensor of shape (len(sentences), feature_dim)
        return torch.arange(len(sentences) * 4).reshape(len(sentences), 4)

class DummyEraser:
    @staticmethod
    def fit(X, one_hot):
        return DummyEraser()
    def __call__(self, X):
        # Erasure returns zeros of same shape
        return torch.zeros_like(X)

class DummyLogisticRegression:
    def __init__(self, max_iter=None):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        # Predict all zeros
        return [0] * X.shape[0]

# Fixture to patch heavy dependencies
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    # Patch load_dataset to return dummy data
    def fake_load_dataset(name, lang, split):
        # Two dummy sentences per language
        return DummyDataset([f"sent_{lang}_1", f"sent_{lang}_2"])
    monkeypatch.setattr("experiments.leace.load_dataset", fake_load_dataset)

    # Patch SentenceTransformer to DummyModel
    monkeypatch.setattr("experiments.leace.SentenceTransformer", lambda model_id: DummyModel())

    # Patch LeaceEraser
    monkeypatch.setattr("experiments.leace.LeaceEraser", DummyEraser)

    # Patch sklearn LogisticRegression
    monkeypatch.setenv("PY_TESTING", "1")
    monkeypatch.setattr("experiments.leace.LogisticRegression", DummyLogisticRegression)

    # Prevent actual file I/O and plotting
    monkeypatch.setattr("experiments.leace.plt.savefig", lambda *args, **kwargs: None)

    # Use a temporary eraser path
    test_eraser = tmp_path / "eraser.pkl"
    monkeypatch.setattr(sys, "argv", ["prog", f"--eraser-path={test_eraser}"])
    yield

# --- Tests ---


def test_load_embeddings_shapes_and_labels():
    langs = ["en", "de"]
    model = DummyModel()
    # Test 'train' split
    X, Y = experiments.leace.load_embeddings(langs, model, split="train")
    # Should have 2 languages * 2 sentences = 4 rows
    assert X.shape == (4, 4)
    assert Y.shape == (4,)
    # Labels: first two zeros, next two ones
    assert Y.tolist() == [0, 0, 1, 1]


def test_main_runs_through(tmp_path, capsys):
    # Ensure eraser_path does not exist -> triggers fit branch
    eraser_file = tmp_path / "eraser.pkl"

    # Modify parse_args to return custom args object
    class Args:
        languages = "en,de"
        embedding_model = "dummy"
        eraser_path = str(eraser_file)

    experiments.leace.parse_args = lambda: Args()

    # Run main should not raise
    experiments.leace.main()
    captured = capsys.readouterr()
    assert "Dev accuracy before erasure" in captured.out
    assert "Fitted and saved new eraser" in captured.out
    assert "Dev accuracy after erasure" in captured.out

    # Run again to hit load branch
    with open(eraser_file, "wb") as f:
        pickle.dump(DummyEraser(), f)
    experiments.leace.main()
    captured2 = capsys.readouterr()
    assert "Loaded existing eraser" in captured2.out
    assert "Dev accuracy after erasure" in captured2.out

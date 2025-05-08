import os
import json
import pickle
import numpy as np
import torch
import pytest
import analysis

# Dummy embedder that returns constant embeddings
class DummyEmbedder:
    def __init__(self, model_id):
        pass

    def encode(self, texts):
        # Return an array of shape (len(texts), 3) with all ones
        ones = np.ones((len(texts), 3))
        for i, num in enumerate(texts):
            ones[i] *= num
        return ones

# Identity eraser module
class IdentityEraser(torch.nn.Module):
    def forward(self, x):
        return x

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch the SentenceTransformer to use our dummy embedder
    monkeypatch.setattr('analysis.baselines.SentenceTransformer', DummyEmbedder)
    # Patch pickle.load to return the identity eraser
    monkeypatch.setattr('pickle.load', lambda fp: IdentityEraser())
    yield


def test_run_creates_correct_data(tmp_path, monkeypatch):
    """
    Test that `run` creates a data.json file with average cosine similarity of 1.0
    when default vs default embeddings are identical.
    """
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create necessary directories
    json_dir = tmp_path / 'outputs' / 'jsons'
    json_dir.mkdir(parents=True)

    # Write the untranslated default JSON
    default = {"en": {"language": {"a": [1, 2], "b": [-1, 1]}}}
    untranslated_fp = json_dir / 'testmodel_untranslated.json'
    with open(untranslated_fp, 'w') as f:
        json.dump(default, f)

    # Call run (inf_model_id -> short='testmodel')
    analysis.baselines.run('models/testmodel', 'dummy_emb', untranslated_fp)

    # Verify that data.json was created and has the correct values
    data_fp = json_dir / 'testmodel_data.json'
    assert data_fp.exists(), "Data file was not created"
    with open(data_fp) as f:
        data = json.load(f)
    print(data)

    # Expect one experiment "default vs default" with value 1.0 for "testmodel"
    assert "default vs default" in data
    val = data["default vs default"].get("testmodel")
    assert val == pytest.approx(0.0, rel=1e-6)

    # Verify that graph images were saved for each attribute
    graph_dir = tmp_path / 'outputs' / 'graphs' / 'specific_plots'
    assert (graph_dir / 'testmodel_a_baseline.png').exists()
    assert (graph_dir / 'testmodel_b_baseline.png').exists()
    
    # Rewrite the untranslated default JSON
    default['fr'] = {}
    untranslated_fp = json_dir / 'testmodel_untranslated.json'
    with open(untranslated_fp, 'w') as f:
        json.dump(default, f)
    # Doesn't contain the proper data for other baselines
    with pytest.raises(FileNotFoundError):
        analysis.baselines.run('models/testmodel', 'dummy_emb', untranslated_fp)


def test_run_with_orthogonal_embeddings(tmp_path, monkeypatch):
    """
    Test that `run` computes average cosine similarity near zero when embeddings are orthogonal.
    """
    # Define an embedder that returns orthogonal embeddings for two texts
    class OrthoEmbedder:
        def __init__(self, model_id):
            pass

        def encode(self, texts):
            # For two texts, return orthogonal 2D vectors
            if len(texts) == 2:
                return np.array([[1.0, 0.0], [0.0, 1.0]])
            # For a single text, return a unit vector
            return np.array([[1.0, 0.0]])

    # Patch dependencies
    monkeypatch.setattr('analysis.baselines.SentenceTransformer', OrthoEmbedder)
    monkeypatch.setattr('pickle.load', lambda fp: IdentityEraser())

    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create necessary directories and default JSON
    json_dir = tmp_path / 'outputs' / 'jsons'
    json_dir.mkdir(parents=True)
    default = {"en": {"language": {"a": ["x", "y"]}}}
    with open(json_dir / 'testmodel_untranslated.json', 'w') as f:
        json.dump(default, f)

    # Run the experiment
    analysis.baselines.run('models/testmodel', 'dummy_emb', json_dir / 'testmodel_untranslated.json')

    # Check data.json for near-zero similarity
    with open(json_dir / 'testmodel_data.json') as f:
        data = json.load(f)

    val = data.get("default vs default", {}).get("testmodel")
    assert val == pytest.approx(0.0, abs=1e-2)

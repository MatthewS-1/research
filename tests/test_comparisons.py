import json
import os
import numpy as np
import analysis.comparisons
import pytest
import analysis
from pathlib import Path
import sys


def test_load_results_success(tmp_path):
    data = {"a": 1, "b": [1, 2, 3]}
    file_path = tmp_path / "data.json"
    file_path.write_text(json.dumps(data))

    result = analysis.comparisons.load_results(str(file_path))
    assert result == data


def test_load_results_file_not_found():
    with pytest.raises(FileNotFoundError):
        analysis.comparisons.load_results("non_existent_file.json")


def test_plot_embeddings_creates_file_and_returns_avg(tmp_path):
    # Create dummy embeddings and labels
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    colors = [0, 1]
    label_map = {"cluster0": 0, "cluster1": 1}
    output_file = tmp_path / "scatter.png"

    avg = analysis.comparisons.plot_embeddings("attr", embeddings, colors, label_map, output_file)
    # Check that file was created
    assert output_file.exists()
    # avg should be list of two means equal to the points themselves
    assert len(avg) == 2
    assert np.allclose(avg[0], [1.0, 0.0])
    assert np.allclose(avg[1], [0.0, 1.0])


def test_plot_matrix_creates_file(tmp_path):
    mat = np.array([[0.1, 0.2], [0.3, 0.4]])
    labels = ["a", "b"]
    output_file = tmp_path / "matrix.png"

    analysis.comparisons.plot_matrix(mat, labels, output_file, "Test Matrix")
    assert output_file.exists()


def test_plot_bar_creates_file(tmp_path):
    vals = [0.5, 0.7, 0.2]
    labs = ["x", "y", "z"]
    output_file = tmp_path / "bar.png"

    analysis.comparisons.plot_bar(vals, labs, output_file)
    assert output_file.exists()


def test_main_runs(monkeypatch, tmp_path):
    dummy_json = {
        "en": {
            "location": {
                "attribute": ["hello"]
            },
            "language": {
                "attribute": ["english-lang"]
            }
        },
        "fr": {
            "location": {
                "attribute": ["french-location"]
            },
            "language": {
                "attribute": ["french-lang"]
            }
        }
    }
    input_dir = tmp_path / "outputs/jsons"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / "model_translated.json"
    input_file.write_text(json.dumps(dummy_json))

    (tmp_path / "outputs/graphs/specific_plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs/graphs/overall_plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs/jsons").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sys, 'argv', [
        'main',
        '--inference-model-id=model',
        f'--input-dir={input_dir}'
    ])

    class DummyModel:
        def encode(self, texts, batch_size=100, show_progress_bar=True):
            print(texts)
            return np.array([
                [1, 1],
                [1, 0],
                [1, 1],
                [-1, 1]
            ])

    monkeypatch.setattr(analysis.comparisons, "SentenceTransformer", lambda _: DummyModel())

    analysis.comparisons.main()

    output_json = tmp_path / "outputs/jsons/model_data.json"
    assert output_json.exists()
    written = json.loads(output_json.read_text())
    print("hello world")
    print("written: ", written)
    assert "fr" in written

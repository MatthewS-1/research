
# BiasFromVec

BiasFromVec is a framework for evaluating biases in language models, particularly those influenced by a user's language or geographic location. This repository outlines the methods used in the BiasFromVec framework and provides implementation details for reproducing and extending the original study. 

# Installation

```
conda env create -f environment.yml 
```

Verify everything works

```
conda activate BiasFromVec && pytest tests/
```

# Usage

If you have a HuggingFace token, you can use BiasFromVec on a HuggingFace model by doing

```

```

and if you have a google translate api and a credentials json, you can add translation by doing

```

```

This uses the original prompts from the paper. If you'd like to create your own, modify experiments/create_prompts.py and script accordingly

The BiasFromVec framework performs the following
  - create prompts for probing for analysis (experiments/create_prompts.py)
  - perform inference on the prompts (experiments/experiment.py)
  - fit LEACE to your languages (experiments/leace.py)
  - translate the format raw outputs (analysis/translation_and_formatting.py)
  - compute cosine similarities for baselines and comparisons (analysis/baselines.py and analysis/comparisons.py)
  - visualize data (analysis/create_graphs.py)

# Documentation

## experiments/experiment.py

This script performs multilingual prompt-based inference using a Hugging Face language model. It:

- Authenticates with the Hugging Face Hub using a user-provided token.
- Loads a structured JSON file containing multilingual prompts.
- Initializes a `text-generation` pipeline with the specified model.
- Constructs a dataset by combining languages, prompt types, and attributes from the prompt file.
- Generates multiple responses per prompt using customizable parameters such as:
  - `num_runs`: number of inference passes (for robustness)
  - `max_new_tokens`: length of generated output
  - `num_return_sequences`: number of completions per prompt
  - `temperature`: sampling randomness
- Repeats inference across all prompts and aggregates results.
- Saves the output in JSON format under `outputs/jsons/`.

> ⚠️ CUDA is required—this script asserts that a GPU is available.

### example

```
python experiments/experiment.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --token your_huggingface_token \
  --prompts_path prompts/llama-prompts.json \
  --num_runs 2 \
  --max_new_tokens 30 \
  --num_return_sequences 2 \
  --temperature 1.0
```
## experiments/leace.py

This script trains and evaluates a LEACE eraser on multilingual embeddings from the Flores-101 dataset. It:

- Loads sentence embeddings for multiple languages using a `SentenceTransformer` model.
- Trains a logistic regression classifier to predict language labels from embeddings (pre-erasure).
- Fits or loads a `LEACE` eraser to remove language-specific information from the embeddings.
- Re-trains and evaluates the classifier on the erased embeddings.
- Evaluates generalization by testing on a held-out `devtest` split.
- Visualizes the effect of erasure using a bar plot comparing:
  - Accuracy before erasure
  - Accuracy after erasure
  - Accuracy on unseen data (devtest)
  - Random chance baseline

> 💾 The eraser is saved to or loaded from a pickle file (`--eraser-path`).

### example

```
python experiments/leace.py \
  --languages eng,deu,fra,ita,por,hin,spa,tha \
  --embedding-model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
  --eraser-path research/models/eraser.pkl
```

## analysis/translation_and_formatting.py

This script processes the generation outputs of a language model by extracting, optionally translating, and saving them in structured JSON files. It:

- Loads model generation outputs from `outputs/jsons/{model_name}_output.json`.
- Reads the corresponding prompt metadata to organize responses by:
  - language
  - prompt type
  - attribute
- Saves the original responses in a nested JSON format (`untranslated.json`).
- If `--translate` is specified:
  - Authenticates using Google Cloud credentials.
  - Translates each generation from its original language to English.
  - Optionally back-translates English generations into other target languages.
  - Saves both translated and back-translated results to appropriately named JSON files.

> 🔐 Translation requires a valid Google Cloud credentials file and access to the Cloud Translation API.

### example

```
python analysis/translation_and_formatting.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --prompts-path prompts/llama-prompts.json \
  --translate \
  --credentials-path ~/.gcloud/translation-creds.json
```

## analysis/baselines.py

This script measures how translation and LEACE erasure affect semantic consistency by computing average cosine similarities on sentence embeddings. It:

- Loads the “untranslated” generations and any existing similarity results from  
  `outputs/jsons/{model_short}_untranslated.json` and  
  `outputs/jsons/{model_short}_data.json`
- Encodes sentences with a `SentenceTransformer` and (optionally) applies a pickled LEACE eraser
- Computes average cosine similarity over 100 random shuffles between:
  - Default vs. Default
  - Default vs. Default translated to each other language **with** LEACE
  - Default vs. Default translated to each other language **and back** (without LEACE)
- Saves per‐attribute bar plots to `outputs/graphs/specific_plots/{model_short}_{attribute}_baseline.png`
- Updates a JSON (`_data.json`) with the aggregated similarity scores

> ⚙️ Assumes the LEACE eraser pickle and the untranslated JSON outputs already exist.

### example

```bash
python experiments/eval_translation_leace.py \
  --inf-model-id Qwen/Qwen2.5-72B-Instruct \
  --emb-model-id sentence-transformers/all-MiniLM-L6-v2 \
  --eraser-path research/models/eraser.pkl
```

## analysis/comparisons.py

This script performs detailed embedding analysis by evaluating how different languages, prompts, and attributes affect the generated sentence embeddings. Specifically, it:

1. Loads results from the provided JSON file (either untranslated or translated based on the LEACE flag).
2. Encodes the text data using the specified `SentenceTransformer` model.
3. Generates visualizations to analyze embeddings, including:
    - Scatter plots of reduced 2D embeddings using PCA
    - Similarity matrices between language-prompt-attribute combinations
    - Bar plots summarizing overall similarity values across different languages
4. Saves the following plots to `graphs/specific_plots` and `graphs/overall_plots`:
    - Scatter plots for each attribute
    - Heatmaps of similarity matrices
    - Bar plots showing the final results
5. Aggregates the similarity data into a JSON file for further analysis.

### Arguments
- `--inference-model-id`: Inference model ID (used to locate JSON outputs).
- `--embedding-model-id`: SentenceTransformer model ID for computing embeddings (default is "Alibaba-NLP/gte-Qwen2-1.5B-instruct").
- `--use-leace`: Flag to apply LEACE erasure on embeddings.
- `--input-dir`: Directory containing input JSON files (default is `outputs/jsons`).

### Example Usage

```bash
python experiments/embedding_analysis.py \
  --inference-model-id Qwen/Qwen2.5-72B-Instruct \
  --embedding-model-id sentence-transformers/all-MiniLM-L6-v2 \
  --use-leace \
  --input-dir outputs/jsons
```

### Visualizations Produced:
- **Scatter Plots**: 2D projections of embeddings for each attribute, grouped by language-prompt-attribute combination.
- **Similarity Matrices**: Heatmaps showing cosine similarities between the averaged embeddings for each language-prompt-attribute pair.
- **Bar Plots**: Overall similarity comparison between the "default" language and others, visualizing the effects of translation and LEACE.

### Notes:
- Assumes the presence of specific input files and directories (`jsons`, `graphs/specific_plots`, `graphs/overall_plots`).
- Can be used to evaluate translation robustness and the impact of LEACE erasure on semantic consistency across various languages and attributes.
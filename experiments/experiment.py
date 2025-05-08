import json
import torch
import argparse
import os
from itertools import product
from huggingface_hub import login
from transformers import pipeline, logging
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def run(
    model_id: str,
    token: str,
    prompts_path: str,
    num_runs: int = 25,
    max_new_tokens: int = 20,
    num_return_sequences: int = 4,
    temperature: float = 1.0
):
    # Set logging verbosity
    logging.set_verbosity_info()

    device = 0 if torch.cuda.is_available() else -1
    assert device == 0, "CUDA is not available"

    print(f"Torch counted {torch.cuda.device_count()} GPU(s)")

    # Login using provided token
    login(token=token)

    # Create the generation pipeline
    generator = pipeline(
        "text-generation", 
        model=model_id, 
        # model_kwargs={"load_in_8bit": True},
        device_map="auto"
    )

    # Load prompts from file
    with open(prompts_path, "r") as json_file:
        prompts = json.load(json_file)

    print("Model is being run on:", next(generator.model.parameters()).device)

    # Prepare dataset from prompt combinations
    data = [
        {
            "language": lang,
            "prompt_type": prompt_type,
            "attribute": attribute,
            "message": message,
        }
        for lang, ptypes in prompts.items()
        for prompt_type, attrs in ptypes.items()
        for attribute, message in attrs.items()
    ]
    dataset = Dataset.from_list(data)
    print("Dataset loaded")

    # Generate responses for each batch
    def generate_response(batch):
        messages = batch["message"]
        response = generator(
            messages,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        batch["response"] = response
        return batch

    all_results = []

    for _ in tqdm(range(num_runs), desc="Running inference"):
        results = dataset.map(generate_response, batched=True, batch_size=2)
        all_results.append(results)

    # Concatenate all result datasets
    final_results = concatenate_datasets(all_results)
    final_results_dict = final_results.to_dict()

    # Save output
    os.makedirs("outputs/jsons/", exist_ok=True)
    output_path = f"outputs/jsons/{model_id.split('/')[-1]}_output.json"
    with open(output_path, "w") as json_file:
        json.dump(final_results_dict, json_file, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multilingual inference")

    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID (e.g., meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--prompts_path", type=str, default="prompts/llama-prompts.json", help="Path to JSON prompts file")
    parser.add_argument("--num_runs", type=int, default=25, help="Number of times to repeat inference")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    parser.add_argument("--num_return_sequences", type=int, default=4, help="Number of return sequences per input")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    run(
        model_id=args.model_id,
        token=args.token,
        prompts_path=args.prompts_path,
        num_runs=args.num_runs,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature
    )

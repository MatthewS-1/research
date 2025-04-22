import argparse
import json
import os
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from tqdm import tqdm
from google.cloud import translate_v2 as translate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract, optionally translate, and save generation outputs."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Full model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)."
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        required=True,
        help="Path to the JSON file containing prompts mapping."
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Whether to perform translation and back-translation."
    )
    parser.add_argument(
        "--credentials-path",
        type=Path,
        default=None,
        help="Path to Google Cloud credentials JSON. Required if --translate is set."
    )
    return parser.parse_args()


def dd():
    return defaultdict(dict)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def setup_translation(credentials_path: Path):
    if not credentials_path or not credentials_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)
    return translate.Client()


@lru_cache(maxsize=None)
def translate_text(client, text: str, source: str, target: str, model: str = 'base'):
    if source == target or not text:
        return text
    result = client.translate(
        text,
        source_language=source,
        target_language=target,
        model=model
    )
    return result.get("translatedText", text)


def main():
    args = parse_args()
    model_name = args.model_id.split("/")[-1]
    # Default paths
    input_path = Path(f"outputs/jsons/{model_name}_output.json")
    output_dir = Path("outputs/jsons")

    # Load data
    results = load_json(input_path)
    prompts = load_json(args.prompts_path)

    # Build untranslated results
    untranslated = defaultdict(dd)
    total = len(results.get('language', []))
    for lang, ptype, attr, resp in tqdm(
        zip(results['language'], results['prompt_type'], results['attribute'], results['response']),
        total=total,
        desc="Building untranslated"
    ):
        untranslated[lang][ptype].setdefault(attr, [])
        for gen in resp:
            texts = gen.get("generated_text", [])
            content = (texts[1].get("content") if len(texts) > 1 else texts[0].get("content", ""))
            untranslated[lang][ptype][attr].append(content)

    save_json(untranslated, output_dir / f"{model_name}_untranslated.json")

    # Translation step
    if args.translate:
        client = setup_translation(args.credentials_path)
        # this is used later for comparing between generations from different prompts
        # Build untranslated results
        untranslated_to_en = defaultdict(dd)
        total = len(results.get('language', []))
        for lang, ptype, attr, resp in tqdm(
            zip(results['language'], results['prompt_type'], results['attribute'], results['response']),
            total=total,
            desc="Building untranslated"
        ):
            untranslated_to_en[lang][ptype].setdefault(attr, [])
            for gen in resp:
                texts = gen.get("generated_text", [])
                content = (texts[1].get("content") if len(texts) > 1 else texts[0].get("content", ""))
                untranslated_to_en[lang][ptype][attr].append(translate_text(client, content, lang, "en"))

        save_json(untranslated_to_en, output_dir / f"{model_name}_translated.json")
        
        # this is used later for creating baselines that measure noise from translation & leace
        for tgt_lang in prompts.keys():
            if tgt_lang == "en":
                continue
            translated_from_en = defaultdict(dd)
            back_translated = defaultdict(dd)
            for lang, ptype, attr, resp in tqdm(
                zip(results['language'], results['prompt_type'], results['attribute'], results['response']),
                total=total,
                desc=f"Translating to {tgt_lang}"
            ):
                if ptype == 'implicit' and lang == 'en':
                    translated_from_en[lang][ptype].setdefault(attr, [])
                    back_translated[lang][ptype].setdefault(attr, [])
                    for gen in resp:
                        texts = gen.get("generated_text", [])
                        content = (texts[1].get("content") if len(texts) > 1 else texts[0].get("content", ""))
                        tr = translate_text(client, content, lang, tgt_lang)
                        translated_from_en[lang][ptype][attr].append(tr)
                        back = translate_text(client, tr, tgt_lang, lang)
                        back_translated[lang][ptype][attr].append(back)

            save_json(translated_from_en, output_dir / f"{model_name}_translated_to_{tgt_lang}.json")
            save_json(back_translated, output_dir / f"{model_name}_translated_{tgt_lang}_and_back.json")

    print("Done.")

if __name__ == '__main__':
    main()

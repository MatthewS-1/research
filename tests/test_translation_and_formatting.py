import sys
import os
import json
import pytest
import analysis.translation_and_formatting as tnf
from pathlib import Path

# --- Fixtures and dummy classes ---
class DummyTranslate:
    def __init__(self):
        pass
    @staticmethod
    def Client():
        return DummyClient()

class DummyClient:
    def __init__(self):
        pass
    def translate(self, text, source_language, target_language, model):
        return {"translatedText": f"[{source_language}->{target_language}]{text}"}

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Make tqdm identity for faster loops
    monkeypatch.setattr(tnf,'tqdm', lambda x, **kwargs: x)
    monkeypatch.setattr(tnf,'translate', DummyTranslate)
    yield

# --- Unit tests ---

def test_parse_args_required(tmp_path):
    sys_argv = sys.argv
    sys.argv = ['prog']
    with pytest.raises(SystemExit):
        tnf.parse_args()
    # Now provide required args
    cred = tmp_path / 'cred.json'
    prompts = tmp_path / 'prompts.json'
    prompts.write_text('{}', encoding='utf-8')
    sys.argv = [
        'prog',
        '--model-id', 'Mod/ID',
        '--prompts-path', str(prompts)
    ]
    args = tnf.parse_args()
    assert args.model_id == 'Mod/ID'
    assert args.prompts_path == Path(str(prompts))
    assert not args.translate


def test_dd_creates_defaultdict_of_dicts():
    d = tnf.dd()
    assert isinstance(d, dict)
    assert 'anything' in d['whatever'].__class__.__name__.lower() or isinstance(d['whatever'], dict)


def test_load_and_save_json(tmp_path):
    data = {'a': 1, 'b': [2,3]}
    fpath = tmp_path / 'sub' / 'file.json'
    # save
    tnf.save_json(data, fpath)
    assert fpath.exists()
    # load
    loaded = tnf.load_json(fpath)
    assert loaded == data


def test_setup_translation_errors(tmp_path):
    # Missing file
    with pytest.raises(FileNotFoundError):
        tnf.setup_translation(tmp_path / 'nope.json')

    # Success
    cred = tmp_path / 'cred.json'
    cred.write_text('{}', encoding='utf-8')
    client = tnf.setup_translation(cred)
    assert isinstance(client, DummyClient)
    assert os.environ['GOOGLE_APPLICATION_CREDENTIALS'] == str(cred)


def test_translate_text_bypass_and_translate(monkeypatch):
    client = DummyClient()
    # No-op cases
    assert tnf.translate_text(client, '', 'en', 'fr') == ''
    assert tnf.translate_text(client, 'x', 'en', 'en') == 'x'
    # Actual translation
    out = tnf.translate_text(client, 'hello', 'en', 'fr', model='base')
    assert out == '[en->fr]hello'


def test_main_without_translate(monkeypatch, tmp_path):
    # Prepare dummy data
    results = {
        'language': ['en'],
        'prompt_type': ['implicit'],
        'attribute': ['attr'],
        'response': [[{'generated_text': [{'content': 'hi'}]}]]
    }
    prompts = {'en': ['p']}
    # Capture loads and saves
    loads = []
    def fake_load(path):
        loads.append(path)
        if 'output.json' in str(path):
            return results
        return prompts
    monkeypatch.setattr(tnf, 'load_json', fake_load)

    saves = []
    def fake_save(data, path):
        saves.append((data, path))
    monkeypatch.setattr(tnf, 'save_json', fake_save)

    # Patch parse_args
    class Args:
        model_id = 'Mod/X'
        prompts_path = Path('dummy')
        translate = False
        credentials_path = None
    monkeypatch.setattr(tnf, 'parse_args', lambda: Args())

    # Run main
    tnf.main()
    # Expect one save for untranslated only
    assert any('_untranslated.json' in str(p) for (_, p) in saves)
    assert all('_translated' not in str(p) for (_, p) in saves)


def test_main_with_translate(monkeypatch, tmp_path):
    # Prepare dummy data
    results = {
        'language': ['en'],
        'prompt_type': ['implicit'],
        'attribute': ['attr'],
        'response': [[{'generated_text': [{'content': 'hi'}]}]]
    }
    prompts = {'en': ['p'], 'fr': []}
    # Monkeypatch loads
    def fake_load(path):
        if 'output.json' in str(path):
            return results
        return prompts
    monkeypatch.setattr(tnf, 'load_json', fake_load)

    # Monkeypatch save_json
    saved = []
    monkeypatch.setattr(tnf, 'save_json', lambda data, path: saved.append(str(path)))

    # Monkeypatch parse_args to enable translate
    cred = tmp_path / 'cred.json'
    cred.write_text('{}', encoding='utf-8')
    class Args:
        model_id = 'Mod/X'
        prompts_path = Path('dummy')
        translate = True
        credentials_path = cred
    monkeypatch.setattr(tnf, 'parse_args', lambda: Args())

    # Monkeypatch setup_translation and translate_text
    monkeypatch.setattr(tnf, 'setup_translation', lambda cp: DummyClient())
    monkeypatch.setattr(tnf, 'translate_text', lambda client, text, s, t, model='base': f"{s}->{t}:{text}")

    # Run main
    tnf.main()
    # Should have saves for untranslated, translated, translated_to_fr.json, translated_fr_and_back.json
    paths = saved
    assert any('untranslated.json' in p for p in paths)
    assert any('translated.json' in p and 'to_' not in p for p in paths)
    assert any('translated_to_fr.json' in p for p in paths)
    assert any('translated_fr_and_back.json' in p for p in paths)

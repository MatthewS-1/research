import json
import pytest
from pathlib import Path
from experiments.experiment import run

def make_dummy_prompts(tmp_path):
    prompts = {
        "en": {
            "location": {
                "favorite author": [
                    {
                        "role": "user",
                        "content": "I'd like to play a game with you. Make a guess without asking any additional questions. I'm from the United States. Guess who my favorite author is."
                    },
                    {
                        "role": "assistant",
                        "content": "Your favorite author is"
                    }
                ],
            }
        }
    }
    prompts_file = tmp_path / "prompts.json"
    with open(prompts_file, "w") as f:
        json.dump(prompts, f)
    return prompts_file, prompts

class DummyGenerator:
    def __init__(self):
        # dummy model with a single parameter having device attribute
        class P: device = 'cpu'
        class M:
            def parameters(self):
                yield P()
        self.model = M()
        # dummy tokenizer with eos_token_id
        class T: eos_token_id = 0
        self.tokenizer = T()

    def __call__(self, messages, **kwargs):
        # simulate returning num_return_sequences generations per message
        n = kwargs.get('num_return_sequences', 2)
        return [[{'generated_text': 'DUMMY_RESPONSE'} for _ in range(n)] for _ in messages]

@pytest.fixture(autouse=True)
def patch_external(monkeypatch, tmp_path, capsys):
    # patch login to no-op
    monkeypatch.setattr('experiments.experiment.login', lambda token: None)
    # patch pipeline to return DummyGenerator
    monkeypatch.setattr('experiments.experiment.pipeline', lambda *args, **kwargs: DummyGenerator())
    # patch torch.cuda
    class Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def device_count(): return 1
    monkeypatch.setattr('torch.cuda', Cuda)
    # run in isolated tmp directory
    monkeypatch.chdir(tmp_path)
    return monkeypatch

def test_data_generation_and_run(tmp_path):
    prompts_file, prompts = make_dummy_prompts(tmp_path)
    # call run with small settings
    run(
        model_id='test-model',
        token='fake-token',
        prompts_path=str(prompts_file),
        num_runs=1,
        max_new_tokens=5,
        num_return_sequences=2,
        temperature=0.5
    )

    out_dir = Path('outputs/jsons')
    out_file = out_dir / 'test-model_output.json'
    assert out_file.exists(), f"Output file {out_file} was not created"

    # load and verify content
    with open(out_file) as f:
        data = json.load(f)
    # Expect keys: language, prompt_type, attribute, message, response
    # Flattened prompts have 2 attributes * 1 prompt_type * 1 language = 2 entries per run
    # num_return_sequences=2, so each response list length is 2
    keys = {k for k in data}
    assert 'response' in keys, "Response key missing in output"
    # Check count: runs=1, entries=2
    print(data['response'][0])
    assert len(data['response'][0]) == 2
    # Verify generated texts match our dummy
    for resp in data['response'][0]:
        assert resp == {'generated_text': 'DUMMY_RESPONSE'}

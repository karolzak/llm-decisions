import csv
import textwrap
from pathlib import Path

import pytest

from edb.settings import get_settings


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch):
    """Isolated configs/data/artifacts dirs with a small synthetic CSV dataset `toy`."""
    configs = tmp_path / "configs"
    for sub in ("datasets", "models", "experiments", "label_catalogs"):
        (configs / sub).mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    labels = ["alpha", "beta", "gamma"]
    for split, n in (("train", 300), ("test", 30)):
        with (src / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "topic", "lab"])
            for i in range(n):
                # every 3 consecutive rows share a topic (group); labels cycle
                w.writerow([f"message {split} {i}", f"t{i // 3}", labels[i % 3]])

    (configs / "datasets" / "toy.yaml").write_text(textwrap.dedent(f"""
        name: toy
        files:
          train: {{path: "{(src / 'train.csv').as_posix()}", format: csv}}
          test: {{path: "{(src / 'test.csv').as_posix()}", format: csv}}
        fields: {{label: lab, group: topic}}
        state: {{template: "Topic {{topic}}: {{text}}"}}
        labels: auto
        task: {{name: pick, instructions: Pick the label.}}
        splits: {{seed: 7, ratios: {{train: 0.6, dev: 0.2, calibration: 0.2}}}}
        label_catalogs: {{short: auto, verbose: toy_verbose}}
    """), encoding="utf-8")
    (configs / "label_catalogs" / "toy_verbose.yaml").write_text(textwrap.dedent("""
        name: toy_verbose
        labels:
          - {value: alpha, description: The first one}
          - {value: beta, description: The second one}
          - {value: gamma, description: The third one}
    """), encoding="utf-8")
    (configs / "models" / "fake-model.yaml").write_text(
        "name: fake-model\nprovider: azure_openai\ndeployment: fake\n"
        "pricing: {input: 1.0, output: 2.0}\n", encoding="utf-8")
    (configs / "experiments" / "fake_exp.yaml").write_text(textwrap.dedent("""
        id: fake_exp
        leaderboards: [product]
        dataset: toy
        label_catalogs: [short, verbose]
        method: {type: fake, model: fake-model, params: {accuracy: 0.8}}
        run: {concurrency: 4, max_retries: 2}
    """), encoding="utf-8")

    monkeypatch.setenv("CONFIGS_DIR", str(configs))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()

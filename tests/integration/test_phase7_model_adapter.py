from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch import nn

from uatf.api import list_model_assets, load_model_asset, predict_model_asset_image, run_training
from uatf.models import MODEL_ADAPTERS, MODEL_REGISTRY, register_model


def test_registered_model_trains_saves_reloads_and_runs_inference(tmp_path: Path) -> None:
    model_name = "phase7_tiny_adapter"

    def build_tiny(output_count: int, pretrained: bool) -> nn.Module:
        assert pretrained is False
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, output_count))

    register_model(
        model_name,
        build_tiny,
        display_name="Phase 7 Tiny Adapter",
        supported_tasks=("classification",),
        supports_pretrained=False,
        input_size=(8, 8),
        input_size_required=True,
        classifier_head="1",
        limitations=("Integration-test adapter.",),
    )
    dataset = tmp_path / "dataset"
    for class_name, color in (("cats", "red"), ("dogs", "blue")):
        for sample_index in range(2):
            image_path = dataset / class_name / f"{sample_index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), color).save(image_path)

    output = tmp_path / "runs"
    configuration = {
        "task": {"type": "classification"},
        "dataset": {
            "format": "image_folder",
            "path": str(dataset),
            "validation_fraction": 0.25,
            "image_size": 8,
            "augmentation": {"enabled": False},
        },
        "model": {"name": model_name, "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "device": "cpu",
            "seed": 7,
        },
        "output": {
            "directory": str(output),
            "save_predictions": False,
            "save_plots": False,
        },
    }

    try:
        experiment = run_training(configuration)
        assets = list_model_assets(output)
        asset = next(item for item in assets if item["architecture"] == model_name)
        loaded = load_model_asset(output, asset["id"])
        prediction = predict_model_asset_image(
            output, asset["id"], dataset / "cats" / "0.png", device="cpu"
        )
        metrics = json.loads((experiment / "metrics.json").read_text(encoding="utf-8"))

        assert isinstance(loaded, nn.Module)
        assert (experiment / "best_model.pth").is_file()
        assert (experiment / "last_model.pth").is_file()
        assert metrics["model"]["adapter"]["display_name"] == "Phase 7 Tiny Adapter"
        assert metrics["evaluation"]["sample_count"] == 1
        assert prediction["task"] == "classification"
        assert prediction["prediction"] in {"cats", "dogs"}
    finally:
        MODEL_REGISTRY.pop(model_name, None)
        MODEL_ADAPTERS.pop(model_name, None)

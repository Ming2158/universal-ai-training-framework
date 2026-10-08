from __future__ import annotations

from pathlib import Path

import pytest

from uatf.config import DEFAULT_CONFIG, load_config, merge_config, validate_config


def test_load_config_requires_a_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  epochs: 2\n", encoding="utf-8")

    assert load_config(config_path) == {"training": {"epochs": 2}}

    config_path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)


def test_load_config_reports_missing_and_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
        load_config(tmp_path / "missing.yaml")

    config_path = tmp_path / "broken.yaml"
    config_path.write_text("task: [broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(config_path)


def test_merge_config_is_recursive_and_does_not_mutate_inputs() -> None:
    defaults = {"training": {"epochs": 10, "amp": {"enabled": False}}}
    user = {"training": {"epochs": 3, "amp": {"enabled": True}}}

    merged = merge_config(defaults, user)

    assert merged == {"training": {"epochs": 3, "amp": {"enabled": True}}}
    assert defaults["training"]["epochs"] == 10
    assert defaults["training"]["amp"]["enabled"] is False
    assert user["training"]["amp"]["enabled"] is True


def test_validate_config_fills_defaults_and_normalizes_task() -> None:
    validated = validate_config(
        {
            "task": {"type": "single_label"},
            "dataset": {"path": "unused-dataset"},
        }
    )

    assert validated["task"]["type"] == "classification"
    assert validated["dataset"]["format"] == "image_folder"
    assert validated["training"]["batch_size"] == DEFAULT_CONFIG["training"]["batch_size"]
    assert validated["training"]["amp"]["enabled"] is False
    assert validated["output"]["directory"] == "outputs"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"unknown": True}, "Unknown option in configuration section 'training'"),
        ({"batch_size": 0}, "training.batch_size must be a positive integer"),
        ({"num_workers": -1}, "training.num_workers must be a non-negative integer"),
        ({"learning_rate": float("inf")}, "training.learning_rate must be finite"),
    ],
)
def test_validate_config_rejects_invalid_training_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(
            {
                "dataset": {"path": "unused-dataset"},
                "training": override,
            }
        )


def test_validate_config_rejects_unknown_sections_and_task_format_mismatch() -> None:
    with pytest.raises(ValueError, match="Unknown configuration section: optimizer"):
        validate_config({"dataset": {"path": "unused-dataset"}, "optimizer": {}})

    with pytest.raises(ValueError, match="Multi-label classification requires"):
        validate_config(
            {
                "task": {"type": "multi_label"},
                "dataset": {"format": "image_folder", "path": "unused-dataset"},
            }
        )


def test_example_config_is_valid() -> None:
    validated = validate_config(load_config(Path("configs/example.yaml")))

    assert validated["task"]["type"] == "multi_label"
    assert validated["dataset"]["format"] == "csv_multilabel"
    assert validated["model"]["name"] == "resnet50"

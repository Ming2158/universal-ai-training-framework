from __future__ import annotations

from pathlib import Path

import pytest

from uatf.config import (
    DEFAULT_CONFIG,
    ConfigurationError,
    dump_config,
    load_config,
    merge_config,
    resolve_config,
    validate_config,
)


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


def test_config_serialization_round_trips_yaml(tmp_path: Path) -> None:
    raw = {"task": {"type": "classification"}, "training": {"epochs": 3}}
    config_path = dump_config(raw, tmp_path / "round-trip.yaml")

    assert load_config(config_path) == raw


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
    assert validated["training"]["optimizer"] == {
        "name": "adam",
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }
    assert validated["dataset"]["augmentation"]["enabled"] is False
    assert validated["training"]["loss"]["name"] == "cross_entropy"
    assert validated["training"]["gradient_accumulation_steps"] == 1
    assert validated["training"]["gradient_clip_norm"] is None
    assert validated["output"]["directory"] == "outputs"
    assert validated["output"]["save_predictions"] is True
    assert validated["output"]["save_plots"] is True


def test_resolve_config_applies_overrides_after_user_values(tmp_path: Path) -> None:
    user = {
        "dataset": {"path": "unused-dataset"},
        "training": {"epochs": 4, "optimizer": {"name": "sgd", "momentum": 0.8}},
        "output": {"directory": str(tmp_path)},
    }

    effective = resolve_config(
        user,
        {"training": {"epochs": 2, "optimizer": {"momentum": 0.9}}},
    )

    assert effective["training"]["epochs"] == 2
    assert effective["training"]["optimizer"]["name"] == "sgd"
    assert effective["training"]["optimizer"]["momentum"] == 0.9
    assert user["training"]["epochs"] == 4
    assert user["training"]["optimizer"]["momentum"] == 0.8


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


def test_validate_config_rejects_task_incompatible_loss_and_model_input_size() -> None:
    with pytest.raises(ValueError, match="Loss 'cross_entropy'.*not compatible with task 'multi_label'"):
        validate_config(
            {
                "task": {"type": "multi_label"},
                "dataset": {"format": "csv_multilabel", "path": "unused-dataset"},
                "training": {"loss": {"name": "cross_entropy"}},
            }
        )

    with pytest.raises(ValueError, match="requires fixed-size image resizing"):
        validate_config(
            {
                "dataset": {"path": "unused-dataset", "resize_enabled": False},
                "model": {"name": "vit"},
            }
        )

    convolution_config = validate_config(
        {
            "dataset": {"path": "unused-dataset", "resize_enabled": False},
            "model": {"name": "resnet50"},
        }
    )
    assert convolution_config["dataset"]["resize_enabled"] is False

    with pytest.raises(ValueError, match="requires dataset.image_size 224 x 224"):
        validate_config(
            {
                "dataset": {"path": "unused-dataset", "image_size": 192},
                "model": {"name": "vit"},
            }
        )


@pytest.mark.parametrize(
    ("dataset", "training", "message"),
    [
        (
            {"augmentation": {"enabled": True, "horizontal_flip_probability": 1.2}},
            {},
            "horizontal_flip_probability must be between 0 and 1",
        ),
        (
            {"augmentation": {"rotation_degrees": 181}},
            {},
            "rotation_degrees cannot exceed 180",
        ),
        (
            {},
            {"gradient_accumulation_steps": 0},
            "gradient_accumulation_steps must be a positive integer",
        ),
        (
            {},
            {"gradient_clip_norm": 0},
            "gradient_clip_norm must be null or a finite number",
        ),
    ],
)
def test_validate_config_rejects_invalid_augmentation_and_gradient_settings(
    dataset: dict[str, object], training: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(
            {
                "dataset": {"path": "unused-dataset", **dataset},
                "training": training,
            }
        )


def test_validate_config_accepts_adamw_and_cosine_scheduler() -> None:
    validated = validate_config(
        {
            "dataset": {"path": "unused-dataset"},
            "training": {
                "epochs": 8,
                "optimizer": {"name": "adamw", "weight_decay": 0.01},
                "scheduler": {"enabled": True, "name": "cosine"},
            },
        }
    )

    assert validated["training"]["optimizer"]["name"] == "adamw"
    assert validated["training"]["scheduler"]["t_max"] == 8


@pytest.mark.parametrize(
    ("training", "message"),
    [
        (
            {"optimizer": {"name": "rmsprop"}},
            "optimizer.name must be 'adam', 'adamw', or 'sgd'",
        ),
        (
            {"scheduler": {"enabled": False, "name": "reduce_on_plateau"}},
            "Unsupported scheduler; currently supported",
        ),
        (
            {"early_stopping": {"enabled": False, "patience": 0}},
            "training.early_stopping.patience must be a positive integer",
        ),
        (
            {"optimizer": {"name": "sgd", "momentum": 0, "nesterov": True}},
            "nesterov requires.*momentum greater than 0",
        ),
        ({"resume_from": "missing-checkpoint.pth"}, "resume_from checkpoint does not exist"),
        ({"device": "cpu", "amp": {"enabled": True}}, "cannot be true.*device is 'cpu'"),
    ],
)
def test_validate_config_rejects_invalid_optimizer_resume_and_device_combinations(
    training: dict[str, object], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        validate_config(
            {
                "dataset": {"path": "unused-dataset"},
                "training": training,
                "output": {"directory": str(tmp_path)},
            }
        )


def test_validate_config_rejects_output_path_that_is_a_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="output.directory points to a file"):
        validate_config(
            {
                "dataset": {"path": "unused-dataset"},
                "output": {"directory": str(output_file)},
            }
        )


def test_multilabel_default_metric_exists_in_evaluation_results(tmp_path: Path) -> None:
    validated = validate_config(
        {
            "task": {"type": "multi_label"},
            "dataset": {"format": "csv_multilabel", "path": "unused-dataset"},
            "output": {"directory": str(tmp_path)},
        }
    )

    assert validated["evaluation"]["primary_metric"] == "micro_f1"


def test_example_config_is_valid() -> None:
    validated = validate_config(load_config(Path("configs/example.yaml")))

    assert validated["task"]["type"] == "multi_label"
    assert validated["dataset"]["format"] == "csv_multilabel"
    assert validated["model"]["name"] == "resnet50"

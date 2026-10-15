from __future__ import annotations

from app import app


def test_workspace_theme_uses_uatf_neutral_surfaces() -> None:
    theme = app.build_theme().to_dict()["theme"]

    assert theme["background_fill_primary"] == "#f4f7f5"
    assert theme["background_fill_secondary"] == "#f4f7f5"
    assert theme["panel_background_fill"] == "#ffffff"
    assert theme["block_background_fill"] == "#ffffff"
    assert theme["neutral_200"] == "#d9e1dc"


def test_workspace_has_separate_pages_for_configuration_and_monitoring() -> None:
    configuration = app.build_app().get_config_file()
    navigation = next(
        component
        for component in configuration["components"]
        if component.get("props", {}).get("elem_id") == "uatf-navigation"
    )
    choices = tuple(choice[1] for choice in navigation["props"]["choices"])
    page_ids = {
        component.get("props", {}).get("elem_id") for component in configuration["components"]
    }

    assert choices == app.WORKSPACE_PAGES
    assert {"page-training", "page-monitor"} <= page_ids


def test_sidebar_brand_navigation_and_version_use_two_divider_boundaries() -> None:
    configuration = app.build_app().get_config_file()
    component_ids = {
        component.get("props", {}).get("elem_id") for component in configuration["components"]
    }

    assert {"uatf-brand", "uatf-navigation", "uatf-version"} <= component_ids
    assert "#uatf-brand {" in app.WORKSPACE_CSS
    assert "border-bottom: 1px solid var(--uatf-border)" in app.WORKSPACE_CSS
    assert "#uatf-version { border-top: 1px solid var(--uatf-border)" in app.WORKSPACE_CSS
    assert ".uatf-brand" not in app.WORKSPACE_CSS


def test_empty_state_components_have_unique_stable_style_targets() -> None:
    configuration = app.build_app().get_config_file()
    empty_state_ids = [
        component.get("props", {}).get("elem_id")
        for component in configuration["components"]
        if str(component.get("props", {}).get("elem_id", "")).startswith(
            "uatf-empty-state-"
        )
    ]

    assert len(empty_state_ids) == 7
    assert set(empty_state_ids) == {
        "uatf-empty-state-dataset",
        "uatf-empty-state-training",
        "uatf-empty-state-experiment",
        "uatf-empty-state-model",
        "uatf-empty-state-evaluation",
        "uatf-empty-state-single-inference",
        "uatf-empty-state-batch-inference",
    }
    assert all(
        "uatf-empty-state"
        not in (component.get("props", {}).get("elem_classes") or [])
        for component in configuration["components"]
    )
    assert '[id^="uatf-empty-state-"]' in app.WORKSPACE_CSS
    assert ".uatf-empty-state" not in app.WORKSPACE_CSS


def test_training_route_opens_monitor_only_after_a_task_id_exists(
    monkeypatch,
) -> None:
    task_result = ("尚未计算",) * 24 + ("task-123",)
    monkeypatch.setattr(app, "start_training", lambda *args: iter([task_result]))

    routed = next(
        app.start_training_for_ui(
            None,
            None,
            None,
            "classification",
            "image_folder",
            0.2,
            42,
            224,
            False,
            "resnet50",
            3,
            8,
            0.001,
            "cpu",
            False,
        )
    )

    assert len(routed) == 25 + 1 + len(app.WORKSPACE_PAGES) + 1
    assert routed[25]["value"] == "训练监控"
    assert [item.get("visible") for item in routed[26 : 26 + len(app.WORKSPACE_PAGES)]] == [
        page == "训练监控" for page in app.WORKSPACE_PAGES
    ]
    assert routed[-1]["interactive"] is False

    validation_failure = ("请先检查数据集",) * 24 + ("",)
    monkeypatch.setattr(
        app,
        "start_training",
        lambda *args: iter([validation_failure]),
    )
    not_routed = next(
        app.start_training_for_ui(
            None,
            None,
            None,
            "classification",
            "image_folder",
            0.2,
            42,
            224,
            False,
            "resnet50",
            3,
            8,
            0.001,
            "cpu",
            False,
        )
    )
    assert not_routed[25].get("value") is None
    assert all(item.get("visible") is None for item in not_routed[26:])


def test_copied_configuration_reports_that_monitor_state_needs_refresh(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app,
        "launch_copied_configuration",
        lambda configuration: ("已启动", "task-copy-1"),
    )

    result = app.launch_copied_configuration_for_ui("task: {}")

    assert result[0] == "已启动"
    assert result[1] == "任务 task-copy-1 已提交给 CLI。点击“刷新任务”读取持久状态。"
    assert result[2] == "task-copy-1"
    assert result[3]["value"] == "训练监控"

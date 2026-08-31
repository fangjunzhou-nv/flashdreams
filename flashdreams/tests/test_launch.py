# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from flashdreams.infra.runner import RunnerConfig
from flashdreams.runtime.demo import RunResult
from flashdreams.scripts import cli
from flashdreams.serving import launch as launch_module
from flashdreams.serving.launch import (
    LaunchModeUnavailableError,
    LaunchOptions,
    ResolvedLaunch,
    available_launch_modes,
    resolve_launch,
)

pytestmark = pytest.mark.ci_cpu


def _runner_config(
    *,
    runner_name: str,
    num_views: int = 1,
    pipeline_name: str | None = None,
) -> RunnerConfig:
    launch_capability = None
    if runner_name.startswith("lingbot-world"):
        launch_capability = "lingbot.launch:LAUNCH_CAPABILITY"
    pipeline = SimpleNamespace(
        name=pipeline_name or runner_name,
        diffusion_model=SimpleNamespace(
            seed=42,
            transformer=SimpleNamespace(num_views=num_views, compile_network=True),
        ),
    )
    return cast(
        RunnerConfig,
        SimpleNamespace(
            runner_name=runner_name,
            launch_capability=launch_capability,
            pipeline=pipeline,
            device="cuda:1",
            pixel_height=480,
            pixel_width=832,
            fps=20,
            output_fps=24,
            example_idx=3,
            postprocess=SimpleNamespace(preset=""),
        ),
    )


def test_lingbot_mp4_launch_validates_manifest_sections(tmp_path: Path) -> None:
    resolved = resolve_launch(
        _runner_config(runner_name="lingbot-world-fast"),
        mode="mp4",
        options=LaunchOptions(
            scenario={"example_idx": 2, "total_blocks": 4},
            output={
                "path": tmp_path / "demo.mp4",
                "fps": 12,
                "stats_path": tmp_path / "stats_demo.json",
            },
        ),
    )

    assert resolved.mode == "mp4"
    assert resolved.summary["output_path"] == tmp_path / "demo.mp4"
    assert resolved.summary["stats_path"] == tmp_path / "stats_demo.json"


def test_lingbot_null_launch_uses_shared_replay_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lingbot.demo import app

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        app, "launch_from_runner", lambda **kwargs: calls.append(kwargs)
    )
    config = _runner_config(runner_name="lingbot-world-fast")

    assert available_launch_modes(config) == ("run", "mp4", "null", "webrtc")
    resolved = resolve_launch(
        config,
        mode="null",
        options=LaunchOptions(
            scenario={"example_idx": 2, "total_blocks": 4},
            output={"fps": 12},
        ),
    )

    assert resolved.mode == "null"
    assert resolved.summary == {
        "runner": "lingbot-world-fast",
        "mode": "null",
        "device": "cuda:1",
    }
    resolved.launch()
    assert calls[0]["config"] is config
    assert calls[0]["mode"] == "null"
    assert calls[0]["scenario"] == {"example_idx": 2, "total_blocks": 4}
    assert calls[0]["output"] == {"fps": 12}


def test_lingbot_null_launch_rejects_output_path() -> None:
    with pytest.raises(ValueError, match="null mode does not write output.path"):
        resolve_launch(
            _runner_config(runner_name="lingbot-world-fast"),
            mode="null",
            options=LaunchOptions(output={"path": "unexpected.mp4"}),
        )


def test_lingbot_launch_rejects_unknown_integration_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported LingBot scenario fields: typo"):
        resolve_launch(
            _runner_config(runner_name="lingbot-world-fast"),
            mode="webrtc",
            options=LaunchOptions(scenario={"typo": True}),
        )


def test_capabilities_extend_launch_without_shared_routing_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCapability:
        def supported_modes(self, config, options):
            del config, options
            return ("webrtc",)

        def resolve(self, config, *, mode, options):
            del config, options
            if mode != "webrtc":
                return None
            return ResolvedLaunch(
                mode="webrtc",
                label="plugin launch",
                launch=lambda: None,
            )

    config = _runner_config(runner_name="third-party-model")
    config.launch_capability = "plugin:capability"
    monkeypatch.setattr(
        launch_module,
        "_load_launch_capability",
        lambda path: _FakeCapability(),
    )

    assert available_launch_modes(config) == ("run", "webrtc")
    assert resolve_launch(config, mode="webrtc").label == "plugin launch"


def test_resolved_launch_calls_integration_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lingbot.demo import app

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        app, "launch_from_runner", lambda **kwargs: calls.append(kwargs)
    )
    config = _runner_config(runner_name="lingbot-world-fast")
    resolved = resolve_launch(
        config,
        mode="webrtc",
        options=LaunchOptions(port=9000),
    )

    resolved.launch()

    assert calls[0]["config"] is config
    assert calls[0]["mode"] == "webrtc"
    assert calls[0]["port"] == 9000


def test_no_instantiate_reports_launch_without_setting_up_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _runner_config(runner_name="lingbot-world-fast")

    cli.main(
        config,
        no_instantiate=True,
        mode="webrtc",
        host="127.0.0.1",
        port=9090,
    )

    output = capsys.readouterr().out
    assert "Available modes: run, mp4, null, webrtc" in output
    assert "Selected launch: LingBot WebRTC server" in output
    assert "'host': '127.0.0.1'" in output


def test_launch_mode_uses_compact_summary_by_default(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lingbot.demo import app

    monkeypatch.setattr(app, "launch_from_runner", lambda **kwargs: None)
    config = _runner_config(runner_name="lingbot-world-fast")

    cli.main(
        config,
        mode="null",
        scenario_overrides={"total_blocks": 1},
    )

    output = capsys.readouterr().out
    assert "Resolved config for" not in output
    assert "Available modes:" not in output
    assert "Resolved runner: 'lingbot-world-fast'" in output
    assert "Selected launch: LingBot null replay" in output
    assert "Scenario: {'total_blocks': 1}" in output


def test_launch_mode_exits_nonzero_for_failed_run_result(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lingbot.demo import app

    def fake_launch_from_runner(**kwargs: object) -> RunResult:
        del kwargs
        return RunResult(status="failed", reason="demo failed")

    monkeypatch.setattr(app, "launch_from_runner", fake_launch_from_runner)
    config = _runner_config(runner_name="lingbot-world-fast")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            config,
            mode="null",
            scenario_overrides={"total_blocks": 1},
        )

    assert exc_info.value.code == 1
    assert "demo failed" in capsys.readouterr().err

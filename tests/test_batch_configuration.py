'''配置、模型准备和批量结果统计的集成契约。'''

import json
from argparse import Namespace
from pathlib import Path

import pytest

import run_agent
from sam3_agent import cli
from sam3_agent.batch import dependencies, runner
from sam3_agent.batch.summary import update_summary
from sam3_agent.config.factory import build_run_config
from sam3_agent.config.schema import AgentConfig
from sam3_agent.tools import Sam3Tool


def config_for(tmp_path, args=None, agent=None, prompt=None):
    '''创建不含真实凭据且不访问服务的测试配置。'''

    return build_run_config(
        {
            "input": {"image_dir": str(tmp_path)},
            "output": {"output_dir": str(tmp_path / "output")},
            "agent": agent or {},
            "prompt": prompt or {},
            "llm": {"model": "fake", "base_url": "http://localhost", "api_key": "test"},
        },
        tmp_path / "config.yaml",
        tmp_path,
        args,
    )


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ({}, False),
        ({"normalize_class_name": False}, False),
        ({"normalize_class_name": True}, True),
    ],
)
def test_prompt_normalization_config_defaults_and_override(tmp_path, prompt, expected):
    '''配置工厂保留默认关闭及显式开关设置。'''

    assert config_for(tmp_path, prompt=prompt).prompt.normalize_class_name is expected


def test_yaml_and_cli_pass_box_limit(tmp_path):
    '''配置默认值和显式命令行覆盖使用同一字段。'''

    config = config_for(tmp_path, agent={"max_box_tasks_per_round": 2})
    assert config.agent.max_box_tasks_per_round == 2
    args = run_agent.build_parser().parse_args(["--max-box-tasks-per-round", "7"])
    config = config_for(tmp_path, args, {"max_box_tasks_per_round": 2})
    assert config.agent.max_box_tasks_per_round == 7
    assert config_for(tmp_path).agent.max_box_tasks_per_round == 4
    assert "mode" not in vars(config.agent)


@pytest.mark.parametrize(
    "budget,boxes", [(-1, 4), (True, 4), (20, 0), (20, False), (20, 1.5)]
)
def test_invalid_limits_fail_before_model_loading(budget, boxes):
    '''拒绝非整数及越界预算，避免消耗模型加载资源。'''

    with pytest.raises(ValueError):
        AgentConfig(budget, False, False, boxes)


def test_batch_backend_enables_interactive_head(tmp_path):
    '''批量入口创建支持框交互的持久后端，但构造时仍不加载。'''

    backend = dependencies.build_sam3_tool(config_for(tmp_path).runtime)
    assert backend.enable_inst_interactivity is True
    assert backend._processor is None


def test_prepare_builds_once_and_rejects_disabled_head(monkeypatch):
    '''准备可以反复调用，已配置错误的后端不静默重建。'''

    class Processor:
        '''只暴露准备阶段使用的交互模型字段。'''

        model = Namespace(inst_interactive_predictor=object())

    calls = []

    def build(backend):
        '''统计实际 processor 创建次数。'''

        calls.append(backend)
        return Processor()

    monkeypatch.setattr(Sam3Tool, "_build_processor", build)
    backend = Sam3Tool(device="cpu", enable_inst_interactivity=True)
    backend.prepare()
    backend.prepare()
    assert calls == [backend]
    disabled = Sam3Tool(device="cpu")
    with pytest.raises(RuntimeError, match="enable_inst_interactivity"):
        disabled.prepare()
    assert calls == [backend]


def test_empty_batch_never_builds_runner(tmp_path, monkeypatch):
    '''空输入仍允许检查配置，并且不加载模型或连接 LLM。'''

    def unexpected_build(*args, **kwargs):
        '''若空输入错误地触发加载则直接使测试失败。'''

        pytest.fail("empty batch must not load a backend")

    monkeypatch.setattr(runner, "build_runner", unexpected_build)
    runner.run_batch(config_for(tmp_path))


def test_single_image_cli_enables_head_and_forwards_limit(tmp_path, monkeypatch):
    '''单图入口自动开启交互头并转发框数量。'''

    built = []
    submitted = []

    def build_backend(**kwargs):
        '''记录构造参数而不加载 SAM。'''

        built.append(kwargs)
        return object()

    def infer(**kwargs):
        '''记录推理参数而不调用服务。'''

        submitted.append(kwargs)
        return {"output_json_path": "result.json"}

    monkeypatch.setattr(cli, "Sam3Tool", build_backend)
    monkeypatch.setattr(cli, "run_single_image_inference", infer)
    cli.main(
        [
            "--image",
            str(tmp_path / "image.png"),
            "--prompt",
            "fish",
            "--model",
            "fake",
            "--base-url",
            "http://localhost",
            "--api-key",
            "test",
            "--max-box-tasks-per-round",
            "6",
        ]
    )
    assert built[0]["enable_inst_interactivity"] is True
    assert submitted[0]["max_box_tasks_per_round"] == 6


def test_partial_summary_is_not_success(tmp_path):
    '''部分输出的状态和调用统计保留，且不计为完整成功。'''

    path = tmp_path / "summary.json"
    update_summary(path, "image.png", False, 2, 0.5, "partial", {"box_calls": 3})
    summary = json.loads(path.read_text())["image.png"]
    assert summary["status"] == "partial" and not summary["success"]
    assert summary["statistics"]["box_calls"] == 3

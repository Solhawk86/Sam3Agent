import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from .loader import resolve_path, section
from .schema import (
    AgentConfig,
    InputConfig,
    LLMConfig,
    OutputConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
)


def cli_or_config(
    args: Optional[Namespace],
    config_section: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """优先读取命令行覆盖值，否则读取 YAML 配置值。"""
    cli_key = key.replace("-", "_")
    cli_value = getattr(args, cli_key, None) if args is not None else None
    return cli_value if cli_value is not None else config_section.get(key, default)


def build_llm_config(llm_config: dict[str, Any]) -> LLMConfig:
    """构建 LLM dataclass，并保留从环境变量兜底读取 key 的能力。"""
    api_key = llm_config.get("api_key")
    api_key_env = llm_config.get("api_key_env", "QWEN_API_KEY")
    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env))

    model = llm_config.get("model") or os.environ.get("QWEN_MODEL")
    base_url = llm_config.get("base_url") or os.environ.get("QWEN_BASE_URL")
    extra_body = llm_config.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("llm.extra_body must be a mapping")
    return LLMConfig(
        provider=llm_config.get("provider", "openai"),
        name=llm_config.get("name") or model,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        max_tokens=llm_config.get("max_tokens", 4096),
        extra_body=extra_body,
    )


def build_run_config(
    raw_config: dict[str, Any],
    config_path: Path,
    workspace_root: Path,
    args: Optional[Namespace] = None,
) -> RunConfig:
    """
    把 YAML 字典和命令行覆盖项合并为结构化运行配置。
    """
    runtime_config = section(raw_config, "runtime")
    llm_config = section(raw_config, "llm")
    input_config = section(raw_config, "input")
    output_config = section(raw_config, "output")
    prompt_config = section(raw_config, "prompt")
    agent_config = section(raw_config, "agent")
    environment = section(raw_config, "environment")

    output_dir = resolve_path(
        cli_or_config(args, output_config, "output_dir"),
        workspace_root,
    )
    if output_dir is None:
        raise ValueError("Missing config: output.output_dir")

    return RunConfig(
        config_path=config_path,
        workspace_root=workspace_root,
        environment=environment,
        runtime=RuntimeConfig(
            sam3_root=resolve_path(runtime_config.get("sam3_root"), workspace_root),
            checkpoint_path=resolve_path(
                runtime_config.get("checkpoint_path"),
                workspace_root,
            ),
            device=runtime_config.get("device"),
            confidence_threshold=runtime_config.get("confidence_threshold", 0.5),
            bpe_path=resolve_path(runtime_config.get("bpe_path"), workspace_root),
            load_from_hf=runtime_config.get("load_from_hf", False),
            chdir_sam3_root=runtime_config.get("chdir_sam3_root", True),
        ),
        llm=build_llm_config(llm_config),
        input=InputConfig(
            image_dir=resolve_path(
                cli_or_config(args, input_config, "image_dir"),
                workspace_root,
            ),
            image_list=resolve_path(
                cli_or_config(args, input_config, "image_list"),
                workspace_root,
            ),
            pattern=cli_or_config(args, input_config, "pattern", "*.jpg"),
            start_index=cli_or_config(args, input_config, "start_index", 0),
            limit=cli_or_config(args, input_config, "limit"),
        ),
        output=OutputConfig(
            output_dir=output_dir,
            final_mask_dir=resolve_path(
                cli_or_config(args, output_config, "final_mask_dir"),
                workspace_root,
            ),
        ),
        prompt=PromptConfig(
            regex=prompt_config.get("regex", r"([A-Za-z_]+?)(\d+)$"),
            replacement=prompt_config.get("replacement", r"\1"),
            underscore_replacement=prompt_config.get("underscore_replacement", ""),
            require_match=prompt_config.get("require_match", True),
        ),
        agent=AgentConfig(
            max_generations=cli_or_config(args, agent_config, "max_generations", 20),
            max_box_tasks_per_round=cli_or_config(
                args, agent_config, "max_box_tasks_per_round", 4
            ),
            debug=cli_or_config(args, agent_config, "debug", False),
            verbose=cli_or_config(args, agent_config, "verbose", False),
        ),
        raw=raw_config,
    )

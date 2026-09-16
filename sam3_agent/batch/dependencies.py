from functools import partial
from typing import Any

from sam3_agent.config.schema import LLMConfig, RunConfig, RuntimeConfig


def get_llm_config(config: LLMConfig) -> dict[str, Any]:
    """把结构化 LLM 配置转换为推理函数需要的字典。"""
    missing = []
    if not config.model:
        missing.append("llm.model")
    if not config.base_url:
        missing.append("llm.base_url")
    if not config.api_key:
        missing.append("llm.api_key or llm.api_key_env")
    if missing:
        raise ValueError("Missing config: " + ", ".join(missing))

    return {
        "provider": config.provider,
        "name": config.name or config.model,
        "model": config.model,
        "base_url": config.base_url,
        "api_key": config.api_key,
        "max_tokens": config.max_tokens,
        "extra_body": config.extra_body,
    }


def build_sam3_tool(config: RuntimeConfig):
    """根据 runtime 配置创建 sam3-agent 内置的 SAM3 分割工具。"""
    from sam3_agent.tools import Sam3Tool

    return Sam3Tool(
        checkpoint_path=(
            str(config.checkpoint_path) if config.checkpoint_path is not None else None
        ),
        device=config.device,
        confidence_threshold=config.confidence_threshold,
        bpe_path=str(config.bpe_path) if config.bpe_path is not None else None,
        load_from_hf=config.load_from_hf,
        enable_inst_interactivity=True,
    )


def build_runner(config: RunConfig, verbose: bool = False):
    """创建 LLM 请求函数、SAM3 工具和单图推理函数依赖。"""
    from sam3_agent.inference import get_result_dir, run_single_image_inference
    from sam3_agent.llm_client import send_generate_request

    llm_config = get_llm_config(config.llm)
    sam_tool = build_sam3_tool(config.runtime)
    request = partial(
        send_generate_request,
        server_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        max_tokens=llm_config["max_tokens"],
        extra_body=llm_config["extra_body"],
        verbose=verbose,
    )
    return llm_config, request, sam_tool, get_result_dir, run_single_image_inference

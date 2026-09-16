# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

'''批量分割 Agent 的公共入口。'''

from typing import Any, Callable

from PIL import Image

from .llm_client import send_generate_request
from .tools.protocol import SingleImageSegmentationBackend


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request: Callable[..., Any] = send_generate_request,
    segmentation_tool: SingleImageSegmentationBackend | None = None,
    max_generations: int = 20,
    output_dir: str = "agent_output",
    llm_log_dir: str | None = None,
    verbose: bool = True,
    max_box_tasks_per_round: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any], Image.Image]:
    '''运行单图批量审核与分割，返回最后上下文、结果及渲染图。'''

    from .segmentation_memory.runner import run_memory_agent

    if segmentation_tool is None:
        raise ValueError("segmentation_tool is required")
    return run_memory_agent(
        image_path=img_path,
        query=initial_text_prompt,
        backend=segmentation_tool,
        request=send_generate_request,
        max_generations=max_generations,
        max_boxes=max_box_tasks_per_round,
        output_dir=output_dir,
        result_dir=llm_log_dir,
        verbose=verbose,
        debug=debug,
    )

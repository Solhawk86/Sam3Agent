'''按请求预算运行批量决策，并将聊天上下文与持久状态分离。'''

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from ..config.schema import validate_agent_limits
from ..llm_client import LLMResponse
from ..tools.protocol import (
    SingleImageSegmentationBackend,
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from ..tools.registration import build_agent_tool_registry
from .messages import build_messages
from .models import SegmentationMemory
from .rendering import render_candidates
from .storage import MemorySession


def _call_tool(
    response: LLMResponse, registry: ToolRegistry, context: ToolContext
) -> ToolResult:
    '''对单个原生调用返回可纠正的 JSON、参数或工具名称错误。'''

    call = response.tool_calls[0]
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError) as error:
        return ToolResult(
            {"status": "invalid_decision", "message": str(error)}, success=False
        )
    if call.name != "advance_segmentation":
        return ToolResult(
            {
                "status": "invalid_decision",
                "message": "Use only advance_segmentation",
            },
            success=False,
        )
    return registry.execute(call.name, context, arguments)


def _prepare_backend(
    backend: SingleImageSegmentationBackend, session: MemorySession
) -> None:
    '''在请求模型前验证分支能力并准备真实后端，测试替身无需加载模型。'''

    start = time.perf_counter()
    try:
        for name in ("segment_phrase", "segment_instance_with_box"):
            if not callable(getattr(backend, name, None)):
                raise TypeError(f"Segmentation backend must implement {name}")
        prepare = getattr(backend, "prepare", None)
        if callable(prepare):
            prepare()
    except Exception as error:
        session.memory.status = "error"
        session.memory.termination_reason = "startup_error"
        session.save_event("startup_error", {"message": str(error)})
        raise
    finally:
        session.memory.statistics.startup_seconds = time.perf_counter() - start
        session.save()


def _request_round(
    request: Callable[..., Any],
    messages: list[dict],
    registry: ToolRegistry,
    context: ToolContext,
    session: MemorySession,
) -> LLMResponse | None:
    '''记录每一次真实 LLM 调用，包括无响应或请求异常。'''

    memory = session.memory
    memory.statistics.llm_requests += 1
    start = time.perf_counter()
    try:
        response = request(
            messages,
            tools=registry.definitions(context),
            tool_choice="required",
            parallel_tool_calls=False,
        )
    except Exception as error:
        memory.status = "partial"
        memory.termination_reason = "llm_error"
        session.save_event("llm_error", {"message": str(error)})
        return None
    finally:
        memory.statistics.llm_seconds += time.perf_counter() - start
    session.save_response(response)
    if response is None:
        memory.status = "partial"
        memory.termination_reason = "llm_no_response"
    elif not isinstance(response, LLMResponse) or len(response.tool_calls) != 1:
        memory.status = "partial"
        memory.termination_reason = "protocol_error"
    else:
        return response
    session.save_event("request_failed", {"reason": memory.termination_reason})
    return None


def run_memory_agent(
    image_path: str,
    query: str,
    backend: SingleImageSegmentationBackend,
    request: Callable[..., Any],
    max_generations: int,
    max_boxes: int,
    output_dir: str,
    result_dir: str | None,
    verbose: bool,
    debug: bool,
) -> tuple[list[dict], dict[str, Any], Image.Image]:
    '''执行唯一的批量记忆流程，预算耗尽时仅导出已接受结果。'''

    validate_agent_limits(max_generations, max_boxes)
    start = time.perf_counter()
    image_path = str(Path(image_path).resolve())
    with Image.open(image_path) as image:
        width, height = image.size
    directory = (
        Path(result_dir)
        if result_dir
        else Path(output_dir) / "result" / Path(image_path).stem
    )
    memory = SegmentationMemory(image_path, query, width, height, uuid4().hex)
    session = MemorySession(directory, memory)
    _prepare_backend(backend, session)
    context = ToolContext(
        image_path,
        query,
        str(session.run_dir / "attempts"),
        memory_session=session,
        verbose=verbose,
    )
    registry = build_agent_tool_registry(backend, max_boxes)
    prompt_path = Path(__file__).parents[1] / "system_prompts" / "system_prompt.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")
    latest_pair = []
    board_path = None
    messages = build_messages(
        system_prompt, memory, max_generations, latest_pair, board_path
    )
    try:
        for round_number in range(1, max_generations + 1):
            messages = build_messages(
                system_prompt,
                memory,
                max_generations - round_number + 1,
                latest_pair,
                board_path,
            )
            memory.round_number = round_number
            session.save_round(messages)
            session.save()
            if verbose:
                print(
                    f"Batch round {round_number}/{max_generations}: {Path(image_path).name}"
                )
            response = _request_round(request, messages, registry, context, session)
            if response is None:
                break
            result = _call_tool(response, registry, context)
            latest_pair = [
                response.as_assistant_message(),
                {
                    "role": "tool",
                    "tool_call_id": response.tool_calls[0].id,
                    "content": result.as_tool_content(),
                },
            ]
            if result.image_path is not None:
                board_path = result.image_path
            if not result.success:
                session.save_event("decision_feedback", result.content)
            if debug:
                session.save_event("debug_tool_response", result.content)
            if result.terminal:
                break
        if memory.status == "running":
            memory.status = "partial"
            memory.termination_reason = "budget_exhausted"
    except Exception as error:
        memory.status = "error"
        memory.termination_reason = "execution_error"
        session.save_event("execution_error", {"message": str(error)})
        raise
    finally:
        session.save()

    render_start = time.perf_counter()
    accepted = [
        item for item in memory.candidates.values() if item.status == "accepted"
    ]
    rendered = render_candidates(memory, accepted)
    memory.statistics.render_seconds += time.perf_counter() - render_start
    messages = build_messages(
        system_prompt,
        memory,
        max_generations - memory.statistics.llm_requests,
        latest_pair,
        board_path,
    )
    memory.statistics.total_seconds = time.perf_counter() - start
    session.save_event(
        "run_finished", {"status": memory.status, "reason": memory.termination_reason}
    )
    outputs = memory.outputs()
    outputs.update(
        {
            "status": memory.status,
            "termination_reason": memory.termination_reason,
            "run_id": memory.run_id,
            "run_dir": str(session.run_dir),
            "statistics": asdict(memory.statistics),
        }
    )
    return messages, outputs, rendered

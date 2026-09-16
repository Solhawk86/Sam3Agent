'''在一个 SAM 实例上顺序执行批次，不发起 LLM 请求。'''

import time
from dataclasses import asdict, dataclass

import torch

from ..tools.protocol import SingleImageSegmentationBackend, ToolContext
from ..tools.segment_instance_with_box import SegmentInstanceWithBoxTool
from ..tools.segment_phrase import SegmentPhraseTool
from .decision import SegmentationTask
from .storage import MemorySession


@dataclass(frozen=True)
class BatchExecutionResult:
    '''汇总本批调用记录和是否需要终止推理。'''

    attempt_ids: list[str]
    fatal_error: bool = False


def is_fatal_backend_error(error: Exception) -> bool:
    '''识别内存不足和 CUDA 状态错误，禁止继续复用受损后端。'''

    return isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or (
        isinstance(error, RuntimeError)
        and any(
            word in str(error).lower()
            for word in ("cuda", "out of memory", "cublas", "cudnn")
        )
    )


def execute_tasks(
    tasks: list[SegmentationTask],
    backend: SingleImageSegmentationBackend,
    context: ToolContext,
    session: MemorySession,
) -> BatchExecutionResult:
    '''隔离单任务上下文，复用精确请求，并保留部分成功结果。'''

    memory = session.memory
    tools = {
        "text": SegmentPhraseTool(backend),
        "box": SegmentInstanceWithBoxTool(backend),
    }
    attempt_ids = []
    fatal = False
    for task in tasks:
        attempt = memory.start_attempt(task.branch, task.arguments)
        attempt_ids.append(attempt.attempt_id)
        if fatal:
            attempt.status = "not_run"
            attempt.error = "A previous task left the backend unsafe to reuse"
            session.save_event("task_not_run", asdict(attempt))
            continue
        key = memory.request_key(task.branch, task.arguments)
        cached_id = memory.request_index.get(key)
        if cached_id is not None:
            cached = memory.attempts[cached_id]
            attempt.status = "reused"
            attempt.mask_ids = list(cached.mask_ids)
            attempt.output_json = cached.output_json
            attempt.reused_attempt_id = cached_id
            memory.statistics.cache_hits += 1
            session.save_event("task_reused", asdict(attempt))
            continue

        temporary = ToolContext(
            image_path=context.image_path,
            initial_text_prompt=context.initial_text_prompt,
            sam_output_dir=str(session.run_dir / "attempts" / attempt.attempt_id),
            verbose=context.verbose,
        )
        session.save_event("task_started", asdict(attempt))
        start = time.perf_counter()
        try:
            synchronize = getattr(backend, "synchronize", None)
            if callable(synchronize):
                synchronize()
            start = time.perf_counter()
            result = tools[task.branch].execute(temporary, task.arguments)
            if callable(synchronize):
                synchronize()
            if not result.success or temporary.current_outputs is None:
                raise ValueError("Segmentation tool returned no structured result")
            memory.add_candidates(attempt, temporary.current_outputs)
            attempt.output_json = temporary.latest_output_json
            attempt.status = "ok"
            memory.request_index[key] = attempt.attempt_id
        except (ValueError, OSError, RuntimeError, MemoryError) as error:
            attempt.status = "error"
            attempt.error = f"{type(error).__name__}: {error}"
            fatal = is_fatal_backend_error(error)
            memory.notable_events.append(
                {"round": memory.round_number, "attempt": asdict(attempt)}
            )
        finally:
            attempt.elapsed_sec = time.perf_counter() - start
            statistics = memory.statistics
            if task.branch == "text":
                statistics.text_calls += 1
                statistics.text_seconds += attempt.elapsed_sec
            else:
                statistics.box_calls += 1
                statistics.box_seconds += attempt.elapsed_sec
        session.save_event("task_completed", asdict(attempt))
    return BatchExecutionResult(attempt_ids, fatal)

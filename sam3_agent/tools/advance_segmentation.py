'''批量审核、分割和结束的唯一 Agent 工具。'''

import time
from dataclasses import asdict

from ..segmentation_memory.decision import validate_decision
from ..segmentation_memory.execution import execute_tasks
from ..segmentation_memory.rendering import render_board
from ..segmentation_memory.review import apply_review
from ..segmentation_memory.schema import advance_schema
from .protocol import (
    BaseAgentTool,
    SingleImageSegmentationBackend,
    ToolContext,
    ToolResult,
)


class AdvanceSegmentationTool(BaseAgentTool):
    '''一次提交旧候选审核和下一批 SAM 任务，不在工具内部调用 LLM。'''

    name = "advance_segmentation"
    description = (
        "Review already visible masks, then optionally segment one text prompt and "
        "multiple pixel boxes, request inspection views, or finish. "
        "Pending masks automatically include close-up views in the returned board. "
        "Review new results and their close-ups together in the next call."
    )

    def __init__(self, backend: SingleImageSegmentationBackend, max_boxes: int = 4):
        '''注入持久化 SAM 后端和每批定位框上限。'''

        self.backend = backend
        self.max_boxes = max_boxes
        self.parameters_schema = advance_schema(max_boxes)

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''先验证整条决策，再更新审核、执行任务并生成统一反馈。'''

        session = context.memory_session
        if session is None:
            raise RuntimeError("advance_segmentation requires a memory session")
        memory = session.memory
        try:
            decision = validate_decision(arguments, memory, self.max_boxes)
        except ValueError as error:
            return ToolResult(
                {"status": "invalid_decision", "message": str(error)}, success=False
            )
        apply_review(memory, decision.changes, decision.review)
        if decision.changes:
            session.save_event("review_applied", {"review": decision.review})

        attempts = []
        if decision.finish:
            memory.status = "success"
            memory.termination_reason = decision.finish_reason
        else:
            batch = execute_tasks(decision.tasks, self.backend, context, session)
            attempts = [asdict(memory.attempts[key]) for key in batch.attempt_ids]
            if batch.fatal_error:
                memory.status = "partial"
                memory.termination_reason = "backend_error"

        start = time.perf_counter()
        board = render_board(memory, decision.inspect_ids)
        board_path = session.rounds_dir / f"round_{memory.round_number:03d}.png"
        board.save(board_path)
        memory.statistics.render_seconds += time.perf_counter() - start
        session.save_event(
            "batch_completed",
            {
                "attempt_ids": [item["attempt_id"] for item in attempts],
                "status": memory.status,
            },
        )
        return ToolResult(
            content={
                "status": memory.status,
                "attempts": attempts,
                "visible_mask_ids": memory.visible_ids,
                "inspection_mask_ids": memory.inspection_ids,
                "termination_reason": memory.termination_reason,
            },
            image_path=str(board_path),
            terminal=memory.status != "running",
            success=memory.status != "partial",
        )

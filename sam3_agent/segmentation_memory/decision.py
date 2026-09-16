'''对整个批次进行无副作用的前置校验。'''

from dataclasses import dataclass
from typing import Any

from ..tools.coordinates import validate_box
from ..tools.protocol import require_exact_arguments
from .models import Branch, SegmentationMemory
from .review import ReviewChange, require_list, validate_review


@dataclass(frozen=True)
class SegmentationTask:
    '''描述一项独立 SAM 请求，参数已规范化。'''

    branch: Branch
    arguments: dict[str, Any]


@dataclass(frozen=True)
class BatchDecision:
    '''保存整条已验证的决策，执行阶段不再解析模型参数。'''

    review: dict[str, Any]
    changes: list[ReviewChange]
    tasks: list[SegmentationTask]
    inspect_ids: list[str]
    finish: bool
    finish_reason: str | None


def validate_decision(
    arguments: dict[str, Any], memory: SegmentationMemory, max_boxes: int
) -> BatchDecision:
    '''验证所有字段及审核后的结束条件，失败时不执行任何操作。'''

    require_exact_arguments(
        "advance_segmentation",
        arguments,
        {
            "review",
            "text_prompt",
            "boxes",
            "inspect_mask_ids",
            "finish",
            "finish_reason",
        },
    )
    changes = validate_review(arguments["review"], memory)
    tasks = []
    text = arguments["text_prompt"]
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text_prompt must be null or a non-empty string")
        tasks.append(SegmentationTask("text", {"text_prompt": text.strip()}))
    boxes = require_list(arguments["boxes"], "boxes")
    if len(boxes) > max_boxes:
        raise ValueError(f"At most {max_boxes} box tasks are allowed")
    for item in boxes:
        require_exact_arguments("box task", item, {"box"})
        box = validate_box(item["box"], memory.width, memory.height)
        tasks.append(SegmentationTask("box", {"box": box}))

    inspect_ids = require_list(arguments["inspect_mask_ids"], "inspect_mask_ids")
    if len(inspect_ids) > 4:
        raise ValueError("At most four masks can be inspected per round")
    if any(
        not isinstance(key, str) or key not in memory.presented_ids
        for key in inspect_ids
    ):
        raise ValueError(
            "Inspection requires IDs present in the current memory summary"
        )
    if len(set(inspect_ids)) != len(inspect_ids):
        raise ValueError("Inspection IDs must be unique")

    finish, reason = arguments["finish"], arguments["finish_reason"]
    if not isinstance(finish, bool):
        raise ValueError("finish must be a boolean")
    if finish:
        if tasks or inspect_ids or reason not in ("complete", "no_target"):
            raise ValueError(
                "Finishing requires no tasks/inspection and a valid finish_reason"
            )
        updated = {change.mask_id: change.status for change in changes}
        statuses = [
            updated.get(key, item.status) for key, item in memory.candidates.items()
        ]
        if "pending" in statuses:
            raise ValueError("Review all pending masks before finishing")
        if (reason == "complete") != ("accepted" in statuses):
            raise ValueError(
                "complete requires accepted masks; no_target requires none"
            )
    elif reason is not None or not (changes or tasks or inspect_ids):
        raise ValueError("Continuing requires an action and a null finish_reason")
    return BatchDecision(
        arguments["review"], changes, tasks, inspect_ids, finish, reason
    )

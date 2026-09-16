'''审核变更的前置验证与原子提交。'''

from dataclasses import dataclass
from typing import Any

from ..tools.protocol import require_exact_arguments
from .models import MaskStatus, SegmentationMemory


@dataclass(frozen=True)
class ReviewChange:
    '''描述一次已经校验的候选状态更新。'''

    mask_id: str
    status: MaskStatus
    reason: str


def require_list(value: Any, field: str) -> list:
    '''校验数组类型，不把字符串或对象误当成集合。'''

    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def require_reason(value: Any) -> str:
    '''要求审核理由是非空文字结论。'''

    if not isinstance(value, str) or not value.strip():
        raise ValueError("review reason must be a non-empty string")
    return value.strip()


def validate_review(review: Any, memory: SegmentationMemory) -> list[ReviewChange]:
    '''构造完整更新列表，发生冲突或未展示候选时不修改状态。'''

    require_exact_arguments("review", review, {"accept", "reject", "replace"})
    changes: dict[str, ReviewChange] = {}

    def add(mask_id: Any, status: MaskStatus, reason: str, allowed: set[str]) -> None:
        '''校验单个变更的来源状态、可见性和互斥关系。'''

        if not isinstance(mask_id, str) or mask_id not in memory.candidates:
            raise ValueError(f"Unknown mask ID: {mask_id!r}")
        if mask_id not in memory.visible_ids:
            raise ValueError(f"Inspect {mask_id} before reviewing this historical mask")
        if mask_id in changes:
            raise ValueError(f"Conflicting or repeated review for {mask_id}")
        if memory.candidates[mask_id].status not in allowed:
            raise ValueError(f"Invalid review transition for {mask_id}")
        changes[mask_id] = ReviewChange(mask_id, status, reason)

    for action, status in (("accept", "accepted"), ("reject", "rejected")):
        for item in require_list(review[action], f"review.{action}"):
            require_exact_arguments(action, item, {"mask_id", "reason"})
            add(
                item["mask_id"],
                status,
                require_reason(item["reason"]),
                {"pending", "accepted", "rejected"},
            )
    for item in require_list(review["replace"], "review.replace"):
        require_exact_arguments(
            "replace", item, {"old_mask_ids", "new_mask_ids", "reason"}
        )
        reason = require_reason(item["reason"])
        old_ids = require_list(item["old_mask_ids"], "old_mask_ids")
        new_ids = require_list(item["new_mask_ids"], "new_mask_ids")
        if not old_ids or not new_ids:
            raise ValueError("Replacement requires non-empty old and new mask IDs")
        for mask_id in old_ids:
            add(mask_id, "superseded", reason, {"accepted"})
        for mask_id in new_ids:
            add(
                mask_id,
                "accepted",
                reason,
                {"pending", "accepted", "rejected", "superseded"},
            )

    accepted_count = sum(
        (changes[key].status if key in changes else item.status) == "accepted"
        for key, item in memory.candidates.items()
    )
    if accepted_count > 100:
        raise ValueError("At most 100 masks can be accepted")
    return list(changes.values())


def apply_review(
    memory: SegmentationMemory, changes: list[ReviewChange], review: dict[str, Any]
) -> None:
    '''仅提交已通过整条决策校验的审核，并保留替换关系。'''

    for change in changes:
        candidate = memory.candidates[change.mask_id]
        candidate.status = change.status
        candidate.reason = change.reason
    if changes:
        event = {"round": memory.round_number, "review": review}
        memory.review_history.append(event)
        if review["reject"] or review["replace"]:
            memory.notable_events.append(event)

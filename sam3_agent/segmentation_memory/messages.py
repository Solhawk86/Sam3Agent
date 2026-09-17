'''从持久状态重建模型上下文，保留完整工具调用配对。'''

import json
from dataclasses import asdict
from typing import Any

from .models import SegmentationMemory


def memory_summary(memory: SegmentationMemory, remaining: int) -> dict[str, Any]:
    '''汇总全部有效候选、待审核项及最近的重要历史。'''

    current_attempts = [
        item
        for item in memory.attempts.values()
        if item.round_number == memory.round_number
    ]
    related_ids = {mask_id for item in current_attempts for mask_id in item.mask_ids}
    shown = [
        item
        for item in memory.candidates.values()
        if item.status in {"accepted", "pending"}
        or item.mask_id in memory.visible_ids
        or item.mask_id in related_ids
    ]
    recent = memory.notable_events[-5:]
    historical_ids = set()
    for event in recent:
        review = event.get("review", {})
        historical_ids.update(item["mask_id"] for item in review.get("reject", []))
        for item in review.get("replace", []):
            historical_ids.update(item["old_mask_ids"] + item["new_mask_ids"])
    memory.presented_ids = sorted({item.mask_id for item in shown} | historical_ids)
    attempts = {item.attempt_id: memory.attempts[item.attempt_id] for item in shown}
    return {
        "image_width": memory.width,
        "image_height": memory.height,
        "box_coordinates": "original image pixels [x1,y1,x2,y2]",
        "query": memory.query,
        "candidates": [item.describe() for item in shown],
        "sources": {
            key: {"branch": item.branch, "arguments": item.arguments}
            for key, item in attempts.items()
        },
        "visible_mask_ids": memory.visible_ids,
        "inspection_mask_ids": memory.inspection_ids,
        "tried_text_prompts": sorted(
            {
                item.arguments["text_prompt"]
                for item in memory.attempts.values()
                if item.branch == "text"
            }
        ),
        "recent_events": recent,
        "batch_attempts": [asdict(item) for item in current_attempts],
        "remaining_llm_requests": remaining,
    }


def build_messages(
    system_prompt: str,
    memory: SegmentationMemory,
    remaining: int,
    latest_pair: list[dict[str, Any]],
    board_path: str | None,
) -> list[dict[str, Any]]:
    '''只保留原图、状态摘要、最近完整调用配对及当前汇总图。'''

    summary = memory_summary(memory, remaining)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": memory.image_path},
                {"type": "text", "text": json.dumps(summary, ensure_ascii=False)},
            ],
        },
        *latest_pair,
    ]
    if board_path is not None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Current accepted/pending masks, automatic close-up views "
                            "for every pending mask, and requested inspection views. "
                            "IDs are stable. Review the supplied close-ups directly; "
                            "do not request the same views again."
                        ),
                    },
                    {"type": "image", "image": board_path},
                ],
            }
        )
    return messages

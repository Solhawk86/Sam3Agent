'''候选审核的状态机、边界条件和整批原子性。'''

import copy

import pytest

from sam3_agent.segmentation_memory.decision import validate_decision
from sam3_agent.segmentation_memory.models import Candidate, SegmentationMemory
from sam3_agent.segmentation_memory.review import apply_review


def memory_with(statuses):
    '''创建已展示候选，用于纯状态机验证，不执行图像计算。'''

    memory = SegmentationMemory("image.png", "fish", 12, 10, "run")
    for index, status in enumerate(statuses, 1):
        key = f"m{index}"
        memory.candidates[key] = Candidate(
            key, "t1", "text", "unused", [0, 0, 1, 1], 0.9, 1, status
        )
    memory.visible_ids = list(memory.candidates)
    memory.presented_ids = list(memory.candidates)
    return memory


def arguments(review):
    '''构造仅审核的完整复合决策。'''

    return {
        "review": review,
        "text_prompt": None,
        "boxes": [],
        "inspect_mask_ids": [],
        "finish": False,
        "finish_reason": None,
    }


def test_many_to_many_replacement_is_atomic():
    '''多目标替换一次提交，旧候选整体退出接受集合。'''

    memory = memory_with(["accepted", "accepted", "pending", "pending"])
    review = {
        "accept": [],
        "reject": [],
        "replace": [
            {
                "old_mask_ids": ["m1", "m2"],
                "new_mask_ids": ["m3", "m4"],
                "reason": "split targets",
            }
        ],
    }
    decision = validate_decision(arguments(review), memory, 4)
    assert memory.candidates["m1"].status == "accepted"
    apply_review(memory, decision.changes, review)
    assert [item.status for item in memory.candidates.values()] == [
        "superseded",
        "superseded",
        "accepted",
        "accepted",
    ]
    assert memory.review_history[0]["review"] == review


def test_superseded_can_only_return_via_explicit_replacement():
    '''已替换版本不能直接接受，但可以通过显式回退替换恢复。'''

    memory = memory_with(["superseded", "accepted"])
    review = {
        "accept": [{"mask_id": "m1", "reason": "rollback"}],
        "reject": [],
        "replace": [],
    }
    with pytest.raises(ValueError):
        validate_decision(arguments(review), memory, 4)
    review = {
        "accept": [],
        "reject": [],
        "replace": [
            {
                "old_mask_ids": ["m2"],
                "new_mask_ids": ["m1"],
                "reason": "rollback",
            }
        ],
    }
    decision = validate_decision(arguments(review), memory, 4)
    apply_review(memory, decision.changes, review)
    assert memory.candidates["m1"].status == "accepted"


def test_mask_id_above_one_hundred_is_valid_but_acceptance_is_bounded():
    '''稳定编号没有一百上限，但同时接受数量不能超过一百。'''

    memory = memory_with(["pending"] * 101)
    one = {
        "accept": [{"mask_id": "m101", "reason": "correct"}],
        "reject": [],
        "replace": [],
    }
    validate_decision(arguments(one), memory, 4)
    all_masks = {
        "accept": [{"mask_id": key, "reason": "correct"} for key in memory.candidates],
        "reject": [],
        "replace": [],
    }
    before = copy.deepcopy(memory)
    with pytest.raises(ValueError, match="100"):
        validate_decision(arguments(all_masks), memory, 4)
    assert memory == before


@pytest.mark.parametrize(
    "box", [[0, 0, float("nan"), 5], [False, 0, 5, 5], [5, 0, 1, 5], [0, 0, 5]]
)
def test_invalid_coordinates_do_not_commit_valid_review(box):
    '''整条决策中框不合法时，前面的合法接受动作仍不得提交。'''

    memory = memory_with(["pending"])
    payload = arguments(
        {
            "accept": [{"mask_id": "m1", "reason": "correct"}],
            "reject": [],
            "replace": [],
        }
    )
    payload["boxes"] = [{"box": box}]
    before = copy.deepcopy(memory)
    with pytest.raises(ValueError):
        validate_decision(payload, memory, 4)
    assert memory == before

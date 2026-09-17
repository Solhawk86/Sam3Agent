'''单图候选状态与调用记录，不保存 GPU 对象。'''

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pycocotools.mask as mask_utils


MaskStatus = Literal["pending", "accepted", "rejected", "superseded"]
Branch = Literal["text", "box"]


@dataclass
class Candidate:
    '''保存稳定编号、原始 RLE 与当前审核结论。'''

    mask_id: str
    attempt_id: str
    branch: Branch
    rle: str
    box: list[float]
    score: float
    area: int
    status: MaskStatus = "pending"
    reason: str = ""

    def describe(self) -> dict[str, Any]:
        '''返回可提供给模型的摘要，排除大体积 RLE。'''

        data = asdict(self)
        del data["rle"]
        data["box_normalized_xywh"] = data.pop("box")
        return data


@dataclass
class Attempt:
    '''记录一次执行或复用，执行错误不会进入成功请求索引。'''

    attempt_id: str
    round_number: int
    branch: Branch
    arguments: dict[str, Any]
    status: str = "running"
    mask_ids: list[str] = field(default_factory=list)
    output_json: str = ""
    reused_attempt_id: str | None = None
    error: str = ""
    elapsed_sec: float = 0.0


@dataclass
class RunStatistics:
    '''分别统计请求、分割、可视化和存储的实际开销。'''

    llm_requests: int = 0
    llm_seconds: float = 0.0
    text_calls: int = 0
    text_seconds: float = 0.0
    box_calls: int = 0
    box_seconds: float = 0.0
    cache_hits: int = 0
    render_seconds: float = 0.0
    storage_seconds: float = 0.0
    startup_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class SegmentationMemory:
    '''维护单次运行的真实状态，聊天摘要不承担状态存储职责。'''

    image_path: str
    query: str
    width: int
    height: int
    run_id: str
    candidates: dict[str, Candidate] = field(default_factory=dict)
    attempts: dict[str, Attempt] = field(default_factory=dict)
    request_index: dict[str, str] = field(default_factory=dict)
    review_history: list[dict[str, Any]] = field(default_factory=list)
    notable_events: list[dict[str, Any]] = field(default_factory=list)
    visible_ids: list[str] = field(default_factory=list)
    inspection_ids: list[str] = field(default_factory=list)
    presented_ids: list[str] = field(default_factory=list)
    statistics: RunStatistics = field(default_factory=RunStatistics)
    round_number: int = 0
    status: str = "running"
    termination_reason: str | None = None

    def start_attempt(self, branch: Branch, arguments: dict[str, Any]) -> Attempt:
        '''在单图内分配稳定调用编号。'''

        attempt = Attempt(
            f"t{len(self.attempts) + 1}", self.round_number, branch, arguments
        )
        self.attempts[attempt.attempt_id] = attempt
        return attempt

    @staticmethod
    def request_key(branch: Branch, arguments: dict[str, Any]) -> str:
        '''用完整规范化参数索引请求，不比较分支间 mask。'''

        return json.dumps([branch, arguments], sort_keys=True, allow_nan=False)

    def add_candidates(self, attempt: Attempt, outputs: dict[str, Any]) -> None:
        '''先验证全部候选，再原子加入结果库，避免半次导入。'''

        if (outputs["orig_img_w"], outputs["orig_img_h"]) != (self.width, self.height):
            raise ValueError("Segmentation output dimensions differ from input image")
        masks = outputs["pred_masks"]
        boxes, scores = outputs["pred_boxes"], outputs["pred_scores"]
        if not (len(masks) == len(boxes) == len(scores)):
            raise ValueError("Segmentation masks, boxes and scores must align")
        new_candidates = []
        for index, (rle, box, score) in enumerate(zip(masks, boxes, scores), 1):
            if not isinstance(rle, str):
                raise ValueError(
                    "Segmentation masks must contain serialized RLE counts"
                )
            candidate = Candidate(
                mask_id=f"m{len(self.candidates) + index}",
                attempt_id=attempt.attempt_id,
                branch=attempt.branch,
                rle=rle,
                box=[float(value) for value in box],
                score=float(score),
                area=int(
                    mask_utils.area({"size": [self.height, self.width], "counts": rle})
                ),
            )
            new_candidates.append(candidate)
        for candidate in new_candidates:
            self.candidates[candidate.mask_id] = candidate
            attempt.mask_ids.append(candidate.mask_id)

    def outputs(self, mask_ids: list[str] | None = None) -> dict[str, Any]:
        '''组装兼容原有格式的结果，默认只导出已接受集合。'''

        candidates = (
            [self.candidates[mask_id] for mask_id in mask_ids]
            if mask_ids is not None
            else [
                item for item in self.candidates.values() if item.status == "accepted"
            ]
        )
        return {
            "original_image_path": self.image_path,
            "orig_img_h": self.height,
            "orig_img_w": self.width,
            "pred_boxes": [item.box for item in candidates],
            "pred_scores": [item.score for item in candidates],
            "pred_masks": [item.rle for item in candidates],
            "mask_ids": [item.mask_id for item in candidates],
        }

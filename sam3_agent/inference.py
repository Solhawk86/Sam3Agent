# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

'''单图推理及完整、部分结果的独立导出。'''

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .agent_core import agent_inference
from .segmentation_memory.storage import write_json_atomic
from .tools.protocol import SingleImageSegmentationBackend


def get_result_dir(output_dir: str, image_path: str) -> str:
    '''按照图片文件名定位单图结果目录。'''

    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, "result", image_basename)


def _build_union_binary_mask(final_output_dict: dict[str, Any]) -> np.ndarray:
    '''合并已选 RLE，输出原图尺寸的零或 255 二值掩码。'''

    height = int(final_output_dict["orig_img_h"])
    width = int(final_output_dict["orig_img_w"])
    merged_mask = np.zeros((height, width), dtype=np.uint8)
    for rle_counts in final_output_dict.get("pred_masks", []):
        decoded = mask_utils.decode({"size": [height, width], "counts": rle_counts})
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        merged_mask |= (decoded > 0).astype(np.uint8)
    return merged_mask * 255


def save_final_binary_mask(
    final_output_dict: dict[str, Any], output_path: str | Path
) -> None:
    '''保存接受集合的二值并集，不引入候选状态以外的筛选。'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_build_union_binary_mask(final_output_dict)).save(output_path)


def run_single_image_inference(
    image_path: str,
    text_prompt: str,
    llm_config: dict[str, Any],
    send_generate_request: Callable[..., Any],
    segmentation_tool: SingleImageSegmentationBackend,
    output_dir: str = "agent_output",
    debug: bool = False,
    max_generations: int = 20,
    final_mask_output_dir: str | None = None,
    verbose: bool = False,
    max_box_tasks_per_round: int = 4,
) -> dict[str, Any]:
    '''执行批量记忆推理，仅将完整结果写入可跳过的完成文件。'''

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    result_dir = Path(get_result_dir(output_dir, image_path))
    result_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = result_dir / "pred.json"
    output_image_path = result_dir / "pred.png"
    history_path = result_dir / "history.json"
    final_mask_path = (
        Path(final_mask_output_dir) / f"{Path(image_path).stem}.png"
        if final_mask_output_dir is not None
        else result_dir / "pred_mask.png"
    )
    if output_json_path.exists():
        return {
            "status": "skipped",
            "result_dir": str(result_dir),
            "output_json_path": str(output_json_path),
            "output_image_path": str(output_image_path),
            "agent_history_path": str(history_path),
            "final_mask_path": str(final_mask_path),
            "llm_name": llm_config["name"],
        }

    history, outputs, rendered = agent_inference(
        image_path,
        text_prompt,
        send_generate_request=send_generate_request,
        segmentation_tool=segmentation_tool,
        output_dir=output_dir,
        debug=debug,
        max_generations=max_generations,
        llm_log_dir=str(result_dir),
        verbose=verbose,
        max_box_tasks_per_round=max_box_tasks_per_round,
    )
    outputs.update({"text_prompt": text_prompt, "image_path": image_path})
    partial = outputs["status"] == "partial"
    if partial:
        output_json_path = result_dir / "partial_pred.json"
        output_image_path = result_dir / "partial_pred.png"
        final_mask_path = result_dir / "partial_mask.png"
        history_path = result_dir / "partial_history.json"

    run_dir = Path(outputs["run_dir"])
    rendered.save(output_image_path)
    rendered.save(run_dir / output_image_path.name)
    save_final_binary_mask(outputs, final_mask_path)
    save_final_binary_mask(
        outputs, run_dir / ("partial_mask.png" if partial else "pred_mask.png")
    )
    write_json_atomic(history_path, history)
    write_json_atomic(run_dir / "history.json", history)
    write_json_atomic(run_dir / output_json_path.name, outputs)
    # 完成 JSON 最后写入，失败的导出不会被下一次运行当作成功跳过。
    write_json_atomic(output_json_path, outputs)
    return {
        "status": outputs["status"],
        "termination_reason": outputs["termination_reason"],
        "result_dir": str(result_dir),
        "run_dir": str(run_dir),
        "output_json_path": str(output_json_path),
        "output_image_path": str(output_image_path),
        "agent_history_path": str(history_path),
        "final_mask_path": str(final_mask_path),
        "llm_name": llm_config["name"],
        "statistics": outputs["statistics"],
    }

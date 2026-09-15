# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import os

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .agent_core import agent_inference
from .tools.protocol import SegmentationTool


def get_result_dir(output_dir, image_path):
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, "result", image_basename)


def _build_union_binary_mask(final_output_dict):
    """Decode the final RLE masks and merge them into a single binary mask."""

    height = int(final_output_dict["orig_img_h"])
    width = int(final_output_dict["orig_img_w"])
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    for rle_counts in final_output_dict.get("pred_masks", []):
        decoded = mask_utils.decode({"size": [height, width], "counts": rle_counts})
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        merged_mask |= (decoded > 0).astype(np.uint8)

    return merged_mask * 255


def save_final_binary_mask(final_output_dict, output_path):
    """Save the union of the final masks as a binary PNG."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    binary_mask = _build_union_binary_mask(final_output_dict)
    Image.fromarray(binary_mask, mode="L").save(output_path)


def run_single_image_inference(
    image_path,
    text_prompt,
    llm_config,
    send_generate_request,
    segmentation_tool: SegmentationTool,
    output_dir="agent_output",
    debug=False,
    max_generations=20,
    final_mask_output_dir=None,
    verbose=False,
):
    """Run inference on a single image with provided prompt"""

    llm_name = llm_config["name"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file names
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = get_result_dir(output_dir, image_path)
    os.makedirs(result_dir, exist_ok=True)

    output_json_path = os.path.join(result_dir, "pred.json")
    output_image_path = os.path.join(result_dir, "pred.png")
    agent_history_path = os.path.join(result_dir, "history.json")
    final_mask_path = (
        os.path.join(final_mask_output_dir, f"{image_basename}.png")
        if final_mask_output_dir is not None
        else None
    )

    # Check if output already exists and skip
    if os.path.exists(output_json_path):
        return {
            "status": "skipped",
            "result_dir": result_dir,
            "output_json_path": output_json_path,
            "output_image_path": output_image_path,
            "agent_history_path": agent_history_path,
            "final_mask_path": final_mask_path,
            "llm_name": llm_name,
        }

    agent_history, final_output_dict, rendered_final_output = agent_inference(
        image_path,
        text_prompt,
        send_generate_request=send_generate_request,
        segmentation_tool=segmentation_tool,
        output_dir=output_dir,
        debug=debug,
        max_generations=max_generations,
        llm_log_dir=result_dir,
        verbose=verbose,
    )

    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_path"] = image_path

    # Save outputs
    json.dump(final_output_dict, open(output_json_path, "w"), indent=4)
    json.dump(agent_history, open(agent_history_path, "w"), indent=4)
    rendered_final_output.save(output_image_path)
    if final_mask_path is not None:
        save_final_binary_mask(final_output_dict, final_mask_path)

    return {
        "status": "success",
        "result_dir": result_dir,
        "output_json_path": output_json_path,
        "output_image_path": output_image_path,
        "agent_history_path": agent_history_path,
        "final_mask_path": final_mask_path,
        "llm_name": llm_name,
    }

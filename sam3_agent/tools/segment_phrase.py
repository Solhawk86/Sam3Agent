'''SAM3 分割后端与 segment_phrase Agent 工具。'''

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from ..helpers.mask_overlap_removal import remove_overlapping_masks
from ..viz import visualize
from .protocol import (
    BaseAgentTool,
    SegmentationResult,
    SegmentationTool,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)


class Sam3Tool:
    '''延迟加载外部 SAM3 包并提供底层短语分割能力。'''

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        bpe_path: Optional[str] = None,
        load_from_hf: bool = False,
    ):
        '''保存 SAM3 加载参数，首次推理时再创建 processor。'''

        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.bpe_path = bpe_path
        self.load_from_hf = load_from_hf
        self._processor = None

    @property
    def processor(self):
        '''按需创建并缓存 SAM3 processor。'''

        if self._processor is None:
            self._processor = self._build_processor()
        return self._processor

    def _build_processor(self):
        '''从外部 SAM3 安装构建图像 processor。'''

        try:
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError(
                "SAM3 is not installed. Install the SAM3 project separately "
                "before using Sam3Tool."
            ) from exc

        bpe_path = self.bpe_path
        if bpe_path is None:
            bpe_path = (
                Path(sam3.__file__).resolve().parent
                / "assets"
                / "bpe_simple_vocab_16e6.txt.gz"
            )

        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=self.load_from_hf,
        )
        return Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    @staticmethod
    def _encode_masks(masks: torch.Tensor):
        '''把二值 mask tensors 编码为可序列化的 COCO RLE。'''

        if masks.numel() == 0:
            return []
        masks_np = masks.detach().to("cpu").numpy().astype(np.uint8)
        encoded = []
        for mask in masks_np:
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            encoded.append(rle["counts"])
        return encoded

    def _infer(self, image_path: str, text_prompt: str):
        '''运行一次 SAM3 推理并转换坐标和 mask 格式。'''

        image = Image.open(image_path)
        orig_width, orig_height = image.size
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        boxes = state["boxes"].detach().to("cpu")
        if boxes.numel() == 0:
            pred_boxes = []
        else:
            pred_boxes = boxes.clone()
            pred_boxes[:, 0] /= orig_width
            pred_boxes[:, 1] /= orig_height
            pred_boxes[:, 2] /= orig_width
            pred_boxes[:, 3] /= orig_height
            pred_boxes[:, 2] -= pred_boxes[:, 0]
            pred_boxes[:, 3] -= pred_boxes[:, 1]
            pred_boxes = pred_boxes.tolist()

        return {
            "orig_img_h": orig_height,
            "orig_img_w": orig_width,
            "pred_boxes": pred_boxes,
            "pred_masks": self._encode_masks(state["masks"].squeeze(1)),
            "pred_scores": state["scores"].detach().to("cpu").tolist(),
        }

    def segment_phrase(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''运行 SAM3 并保存结构化 JSON 和渲染结果。'''

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_dir = Path(output_dir) / Path(image_path).stem
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt = text_prompt.replace("/", "_").replace(os.sep, "_")
        json_path = image_dir / f"{safe_prompt}.json"
        image_path_out = image_dir / f"{safe_prompt}.png"

        if verbose:
            print(f"Running SAM3 for {image_path!r} with prompt {text_prompt!r}")

        data = self._infer(image_path, text_prompt)
        data = remove_overlapping_masks(data)
        data = {
            "original_image_path": image_path,
            "output_image_path": str(image_path_out),
            **data,
        }

        order = sorted(
            range(len(data["pred_scores"])),
            key=lambda index: data["pred_scores"][index],
            reverse=True,
        )
        data["pred_scores"] = [data["pred_scores"][i] for i in order]
        data["pred_boxes"] = [data["pred_boxes"][i] for i in order]
        data["pred_masks"] = [data["pred_masks"][i] for i in order]

        valid = [i for i, mask in enumerate(data["pred_masks"]) if len(mask) > 4]
        data["pred_masks"] = [data["pred_masks"][i] for i in valid]
        data["pred_boxes"] = [data["pred_boxes"][i] for i in valid]
        data["pred_scores"] = [data["pred_scores"][i] for i in valid]

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        visualize(data).save(image_path_out)
        return SegmentationResult(str(json_path), str(image_path_out), data)


class SegmentPhraseTool(BaseAgentTool):
    '''把底层分割器封装成原生 function-call 工具。'''

    name = "segment_phrase"
    description = (
        "Use SAM3 to segment all instances matching one short, simple noun phrase "
        "in the raw input image. A new call replaces all previous masks."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text_prompt": {
                "type": "string",
                "minLength": 1,
                "description": "A short and simple noun phrase, such as 'brown bag'.",
            }
        },
        "required": ["text_prompt"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_tool: SegmentationTool):
        '''注入实际执行 SAM3 推理的底层分割器。'''

        self.segmentation_tool = segmentation_tool

    def execute(
        self,
        context: ToolContext,
        arguments: dict,
    ) -> ToolResult:
        '''校验短语、执行分割并更新当前候选 masks。'''

        require_exact_arguments(self.name, arguments, {"text_prompt"})
        text_prompt = arguments["text_prompt"]
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("segment_phrase.text_prompt must be a non-empty string")
        text_prompt = text_prompt.strip()

        if text_prompt in context.used_text_prompts:
            return ToolResult(
                content={
                    "status": "error",
                    "error": "duplicate_text_prompt",
                    "message": (
                        f"The text_prompt {text_prompt!r} was already used. Call "
                        "segment_phrase again with a different noun phrase."
                    ),
                    "used_text_prompts": sorted(context.used_text_prompts),
                },
                success=False,
            )

        context.used_text_prompts.add(text_prompt)
        result = self.segmentation_tool.segment_phrase(
            image_path=context.image_path,
            text_prompt=text_prompt,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        context.latest_output_json = result.json_path
        context.latest_text_prompt = text_prompt
        context.current_outputs = result.data
        mask_count = len(result.data["pred_masks"])

        if mask_count == 0:
            message = (
                f"No masks were generated for {text_prompt!r}. Try a different, "
                "more general, or more creative simple noun phrase."
            )
            image_path = None
        else:
            message = (
                f"Generated {mask_count} available masks for {text_prompt!r}. "
                "They are numbered in the attached image; compare every mask with "
                f"the raw image and the original query {context.initial_text_prompt!r}."
            )
            image_path = result.image_path

        return ToolResult(
            content={
                "status": "ok",
                "text_prompt": text_prompt,
                "mask_count": mask_count,
                "message": message,
            },
            image_path=image_path,
        )

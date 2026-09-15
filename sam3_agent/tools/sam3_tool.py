"""In-process SAM3 segmentation tool adapter."""

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
from .protocol import SegmentationResult


class Sam3Tool:
    """Lazy-loading adapter around the externally installed SAM3 package."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        bpe_path: Optional[str] = None,
        load_from_hf: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.bpe_path = bpe_path
        self.load_from_hf = load_from_hf
        self._processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self._build_processor()
        return self._processor

    def _build_processor(self):
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
        """Run SAM3 and persist the JSON and rendered result."""
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

        valid = [
            i for i, mask in enumerate(data["pred_masks"]) if len(mask) > 4
        ]
        data["pred_masks"] = [data["pred_masks"][i] for i in valid]
        data["pred_boxes"] = [data["pred_boxes"][i] for i in valid]
        data["pred_scores"] = [data["pred_scores"][i] for i in valid]

        with json_path.open("w") as handle:
            json.dump(data, handle, indent=2)
        visualize(data).save(image_path_out)
        return SegmentationResult(str(json_path), str(image_path_out), data)

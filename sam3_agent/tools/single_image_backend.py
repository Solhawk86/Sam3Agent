'''SAM3 七种单图分割模式共用的延迟加载后端。'''

import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from ..helpers.mask_overlap_removal import remove_overlapping_masks
from ..viz import visualize
from .protocol import SegmentationResult


class Sam3Tool:
    '''延迟加载外部 SAM3 包并提供七种单图分割能力。'''

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        bpe_path: Optional[str] = None,
        load_from_hf: bool = False,
        enable_inst_interactivity: bool = False,
    ):
        '''保存 SAM3 加载参数，首次推理时再创建 processor。'''

        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.bpe_path = bpe_path
        self.load_from_hf = load_from_hf
        self.enable_inst_interactivity = enable_inst_interactivity
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
            enable_inst_interactivity=self.enable_inst_interactivity,
        )
        return Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def _autocast_context(self):
        '''根据运行设备返回适用的自动混合精度上下文。'''

        if str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _open_image(image_path: str) -> Image.Image:
        '''校验路径并读取为独立的 RGB 图片对象。'''

        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with Image.open(image_path) as source:
            return source.convert("RGB")

    @staticmethod
    def _validate_number(value: Any, field_name: str) -> float:
        '''校验坐标值为有限实数。'''

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must contain only numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return number

    @classmethod
    def _validate_box(
        cls,
        box: Any,
        width: int,
        height: int,
        field_name: str = "box",
    ) -> list[float]:
        '''校验像素 XYXY 框并返回浮点坐标。'''

        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{field_name} must be a four-number [x1, y1, x2, y2] array")
        values = [cls._validate_number(value, field_name) for value in box]
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{field_name} must satisfy 0 <= x1 < x2 <= {width} and "
                f"0 <= y1 < y2 <= {height}"
            )
        return values

    @classmethod
    def _validate_points(
        cls,
        points: Any,
        width: int,
        height: int,
        field_name: str = "points",
    ) -> list[list[float]]:
        '''校验非空像素点数组并返回浮点坐标。'''

        if not isinstance(points, list) or not points:
            raise ValueError(f"{field_name} must be a non-empty array")
        if len(points) > 100:
            raise ValueError(f"{field_name} cannot contain more than 100 points")
        validated = []
        for index, point in enumerate(points):
            point_name = f"{field_name}[{index}]"
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{point_name} must be a two-number [x, y] array")
            x = cls._validate_number(point[0], point_name)
            y = cls._validate_number(point[1], point_name)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"{point_name} must be inside image bounds {width}x{height}"
                )
            validated.append([x, y])
        return validated

    @classmethod
    def _validate_examples(
        cls,
        examples: Any,
        width: int,
        height: int,
        require_positive: bool,
    ) -> list[dict[str, Any]]:
        '''校验正负视觉样例对象及其像素框。'''

        if not isinstance(examples, list) or not examples:
            raise ValueError("examples must be a non-empty array")
        if len(examples) > 100:
            raise ValueError("examples cannot contain more than 100 items")
        validated = []
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or set(example) != {"box", "label"}:
                raise ValueError(
                    f"examples[{index}] must contain exactly 'box' and 'label'"
                )
            label = example["label"]
            if label not in {"positive", "negative"}:
                raise ValueError(
                    f"examples[{index}].label must be 'positive' or 'negative'"
                )
            validated.append(
                {
                    "box": cls._validate_box(
                        example["box"],
                        width,
                        height,
                        f"examples[{index}].box",
                    ),
                    "label": label,
                }
            )
        if require_positive and not any(
            example["label"] == "positive" for example in validated
        ):
            raise ValueError("segment_visual_examples requires a positive example")
        return validated

    @classmethod
    def _split_labeled_points(
        cls,
        points: Any,
        width: int,
        height: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        '''校验带标签点并拆分为前景点和背景点。'''

        if not isinstance(points, list) or not points:
            raise ValueError("points must be a non-empty array")
        if len(points) > 100:
            raise ValueError("points cannot contain more than 100 items")
        foreground_points = []
        background_points = []
        for index, item in enumerate(points):
            if not isinstance(item, dict) or set(item) != {"point", "label"}:
                raise ValueError(
                    f"points[{index}] must contain exactly 'point' and 'label'"
                )
            label = item["label"]
            if label not in {"foreground", "background"}:
                raise ValueError(
                    f"points[{index}].label must be 'foreground' or 'background'"
                )
            point = cls._validate_points(
                [item["point"]],
                width,
                height,
                f"points[{index}].point",
            )[0]
            if label == "foreground":
                foreground_points.append(point)
            else:
                background_points.append(point)
        return foreground_points, background_points

    @staticmethod
    def _xyxy_to_normalized_xywh(
        boxes: Any,
        width: int,
        height: int,
    ) -> list[list[float]]:
        '''把像素 XYXY 框转换为归一化 XYWH。'''

        if torch.is_tensor(boxes):
            boxes_np = boxes.detach().to("cpu").float().numpy()
        else:
            boxes_np = np.asarray(boxes, dtype=np.float32)
        if boxes_np.size == 0:
            return []
        boxes_np = boxes_np.reshape(-1, 4).copy()
        boxes_np[:, 2] -= boxes_np[:, 0]
        boxes_np[:, 3] -= boxes_np[:, 1]
        boxes_np[:, [0, 2]] /= width
        boxes_np[:, [1, 3]] /= height
        return boxes_np.tolist()

    @staticmethod
    def _encode_masks(masks: Any) -> list[str]:
        '''把二值 masks 编码为可序列化的 COCO RLE counts。'''

        if torch.is_tensor(masks):
            masks_np = masks.detach().to("cpu").numpy()
        else:
            masks_np = np.asarray(masks)
        if masks_np.size == 0:
            return []
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        encoded = []
        for mask in masks_np:
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            counts = rle["counts"]
            encoded.append(
                counts.decode("utf-8") if isinstance(counts, bytes) else str(counts)
            )
        return encoded

    @staticmethod
    def _sort_and_filter_concept_data(data: dict[str, Any]) -> dict[str, Any]:
        '''按分数排序概念结果并移除无效 RLE。'''

        order = sorted(
            range(len(data["pred_scores"])),
            key=lambda index: data["pred_scores"][index],
            reverse=True,
        )
        data["pred_scores"] = [data["pred_scores"][index] for index in order]
        data["pred_boxes"] = [data["pred_boxes"][index] for index in order]
        data["pred_masks"] = [data["pred_masks"][index] for index in order]
        valid = [
            index for index, mask in enumerate(data["pred_masks"]) if len(mask) > 4
        ]
        data["pred_scores"] = [data["pred_scores"][index] for index in valid]
        data["pred_boxes"] = [data["pred_boxes"][index] for index in valid]
        data["pred_masks"] = [data["pred_masks"][index] for index in valid]
        return data

    @staticmethod
    def _artifact_stem(mode: str, payload: dict[str, Any]) -> str:
        '''根据规范化参数生成稳定且紧凑的产物名称。'''

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        return f"{mode}_{digest}"

    def _unique_artifact_stem(
        self,
        image_path: str,
        output_dir: str,
        mode: str,
        payload: dict[str, Any],
    ) -> str:
        '''用参数哈希和递增后缀避免覆盖已有产物。'''

        base_stem = self._artifact_stem(mode, payload)
        image_dir = self._image_output_dir(image_path, output_dir)
        candidate = base_stem
        suffix = 1
        while (
            (image_dir / f"{candidate}.json").exists()
            or (image_dir / f"{candidate}.png").exists()
            or any((image_dir / "refinement").glob(f"{candidate}_candidate_*.npz"))
        ):
            suffix += 1
            candidate = f"{base_stem}_{suffix:03d}"
        return candidate

    @staticmethod
    def _persist_result(
        image_path: str,
        output_dir: str,
        artifact_stem: str,
        data: dict[str, Any],
    ) -> SegmentationResult:
        '''保存结构化 JSON 和带编号 mask 的渲染图。'''

        image_dir = Path(output_dir) / Path(image_path).stem
        image_dir.mkdir(parents=True, exist_ok=True)
        json_path = image_dir / f"{artifact_stem}.json"
        rendered_path = image_dir / f"{artifact_stem}.png"
        persisted = {
            "original_image_path": image_path,
            "output_image_path": str(rendered_path),
            **data,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(persisted, handle, indent=2, ensure_ascii=False)
        visualize(persisted).save(rendered_path)
        return SegmentationResult(str(json_path), str(rendered_path), persisted)

    def _run_concept(
        self,
        image_path: str,
        text_prompt: Optional[str],
        examples: list[dict[str, Any]],
        require_positive: bool,
        allow_empty_examples: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        '''执行文本、视觉样例或二者联合的概念分割。'''

        image = self._open_image(image_path)
        width, height = image.size
        if allow_empty_examples and not examples:
            validated_examples = []
        else:
            validated_examples = self._validate_examples(
                examples,
                width,
                height,
                require_positive,
            )
        with torch.inference_mode(), self._autocast_context():
            state = self.processor.set_image(image)
            if text_prompt is not None:
                state = self.processor.set_text_prompt(
                    state=state,
                    prompt=text_prompt,
                )
            for example in validated_examples:
                x1, y1, x2, y2 = example["box"]
                normalized_box = [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
                state = self.processor.add_geometric_prompt(
                    box=normalized_box,
                    label=example["label"] == "positive",
                    state=state,
                )

        masks = state["masks"]
        if torch.is_tensor(masks) and masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        scores = state["scores"].detach().to("cpu").reshape(-1).tolist()
        data = {
            "orig_img_h": height,
            "orig_img_w": width,
            "pred_boxes": self._xyxy_to_normalized_xywh(
                state["boxes"],
                width,
                height,
            ),
            "pred_masks": self._encode_masks(masks),
            "pred_scores": scores,
        }
        data = remove_overlapping_masks(data)
        return self._sort_and_filter_concept_data(data), validated_examples

    def segment_phrase(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据文本概念运行 SAM3 并保存结果。'''

        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        text_prompt = text_prompt.strip()
        if verbose:
            print(f"Running SAM3 for {image_path!r} with prompt {text_prompt!r}")
        data, _ = self._run_concept(
            image_path,
            text_prompt,
            [],
            False,
            allow_empty_examples=True,
        )
        safe_prompt = text_prompt.replace("/", "_").replace(os.sep, "_")
        return self._persist_result(image_path, output_dir, safe_prompt, data)

    def segment_visual_examples(
        self,
        image_path: str,
        examples: list[dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据正负视觉样例框执行概念分割并保存结果。'''

        if verbose:
            print(f"Running SAM3 visual examples for {image_path!r}")
        data, validated_examples = self._run_concept(
            image_path,
            None,
            examples,
            True,
        )
        data["prediction_mode"] = "visual_examples"
        data["prompt_data"] = {"examples": validated_examples}
        stem = self._unique_artifact_stem(
            image_path,
            output_dir,
            "visual_examples",
            data["prompt_data"],
        )
        return self._persist_result(image_path, output_dir, stem, data)

    def segment_phrase_with_visual_examples(
        self,
        image_path: str,
        text_prompt: str,
        examples: list[dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''联合文本和正负视觉样例框执行概念分割并保存结果。'''

        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        text_prompt = text_prompt.strip()
        if verbose:
            print(
                f"Running SAM3 for {image_path!r} with text and visual examples"
            )
        data, validated_examples = self._run_concept(
            image_path,
            text_prompt,
            examples,
            False,
        )
        data["prediction_mode"] = "text_with_visual_examples"
        data["prompt_data"] = {
            "text_prompt": text_prompt,
            "examples": validated_examples,
        }
        stem = self._unique_artifact_stem(
            image_path,
            output_dir,
            "text_visual_examples",
            data["prompt_data"],
        )
        return self._persist_result(image_path, output_dir, stem, data)

    def _interactive_model(self):
        '''返回已启用交互头的 SAM3 模型并提供明确错误。'''

        model = self.processor.model
        if getattr(model, "inst_interactive_predictor", None) is None:
            raise RuntimeError(
                "Interactive instance segmentation requires Sam3Tool("
                "enable_inst_interactivity=True)"
            )
        return model

    @staticmethod
    def _boxes_from_masks(
        masks: np.ndarray,
        width: int,
        height: int,
    ) -> list[list[float]]:
        '''从二值 masks 计算归一化 XYWH 紧致框。'''

        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                continue
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1
            boxes.append(
                [
                    x1 / width,
                    y1 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
            )
        return boxes

    def _predict_interactive(
        self,
        image_path: str,
        foreground_points: list[list[float]],
        background_points: list[list[float]],
        box: Optional[list[float]],
        mask_input: Optional[np.ndarray],
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        '''运行一次交互式实例预测并保持候选与 logits 对齐。'''

        image = self._open_image(image_path)
        width, height = image.size
        validated_foreground = (
            self._validate_points(
                foreground_points,
                width,
                height,
                "foreground_points",
            )
            if foreground_points
            else []
        )
        validated_background = (
            self._validate_points(
                background_points,
                width,
                height,
                "background_points",
            )
            if background_points
            else []
        )
        validated_box = (
            self._validate_box(box, width, height) if box is not None else None
        )
        if not validated_foreground and not validated_background and validated_box is None:
            raise ValueError("At least one point or box prompt is required")

        point_coords = validated_foreground + validated_background
        point_labels = [1] * len(validated_foreground) + [0] * len(
            validated_background
        )
        model = self._interactive_model()
        with torch.inference_mode(), self._autocast_context():
            state = self.processor.set_image(image)
            masks, scores, low_res_masks = model.predict_inst(
                state,
                point_coords=(
                    np.asarray(point_coords, dtype=np.float32)
                    if point_coords
                    else None
                ),
                point_labels=(
                    np.asarray(point_labels, dtype=np.int32)
                    if point_labels
                    else None
                ),
                box=(
                    np.asarray(validated_box, dtype=np.float32)
                    if validated_box is not None
                    else None
                ),
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=False,
                normalize_coords=True,
            )

        masks_np = np.asarray(masks)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        low_res_np = np.asarray(low_res_masks, dtype=np.float32)
        if low_res_np.ndim == 2:
            low_res_np = low_res_np[None, ...]
        if masks_np.ndim != 3 or masks_np.shape[-2:] != (height, width):
            raise ValueError("SAM3 returned interactive masks with an invalid shape")
        if len(masks_np) != len(scores_np) or len(masks_np) != len(low_res_np):
            raise ValueError("SAM3 returned misaligned masks, scores, and logits")

        order = np.argsort(-scores_np, kind="stable")
        masks_np = masks_np[order] > 0
        scores_np = scores_np[order]
        low_res_np = low_res_np[order]
        non_empty = np.asarray([bool(mask.any()) for mask in masks_np])
        return (
            masks_np[non_empty],
            scores_np[non_empty],
            low_res_np[non_empty],
            width,
            height,
        )

    @staticmethod
    def _image_output_dir(image_path: str, output_dir: str) -> Path:
        '''返回当前图片专属的输出目录。'''

        return Path(output_dir) / Path(image_path).stem

    def _save_refinement_handles(
        self,
        image_path: str,
        output_dir: str,
        artifact_stem: str,
        low_res_masks: np.ndarray,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        '''逐候选保存安全可加载的低分辨率 logits 句柄。'''

        handle_dir = self._image_output_dir(image_path, output_dir) / "refinement"
        handle_dir.mkdir(parents=True, exist_ok=True)
        handles = []
        serialized_metadata = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        for index, logits in enumerate(low_res_masks):
            handle_path = handle_dir / (
                f"{artifact_stem}_candidate_{index + 1:03d}.npz"
            )
            np.savez_compressed(
                handle_path,
                mask_logits=np.asarray(logits, dtype=np.float32)[None, ...],
                metadata=np.asarray(serialized_metadata),
            )
            handles.append(
                {
                    "mask_number": index + 1,
                    "refinement_handle": str(handle_path),
                }
            )
        return handles

    def _load_refinement_handle(
        self,
        image_path: str,
        output_dir: str,
        refinement_handle: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        '''安全加载并验证当前图片对应的 refinement handle。'''

        if not isinstance(refinement_handle, str) or not refinement_handle.strip():
            raise ValueError("refinement_handle must be a non-empty string")
        allowed_dir = self._image_output_dir(image_path, output_dir).resolve()
        handle_path = Path(refinement_handle.strip()).expanduser().resolve()
        try:
            handle_path.relative_to(allowed_dir)
        except ValueError as exc:
            raise ValueError(
                "refinement_handle must be inside the current image output directory"
            ) from exc
        if handle_path.suffix != ".npz" or not handle_path.is_file():
            raise ValueError("refinement_handle does not reference a valid .npz file")
        try:
            with np.load(handle_path, allow_pickle=False) as archive:
                if set(archive.files) != {"mask_logits", "metadata"}:
                    raise ValueError("refinement_handle has unexpected fields")
                mask_logits = np.asarray(archive["mask_logits"], dtype=np.float32)
                metadata_value = archive["metadata"]
                if metadata_value.shape != ():
                    raise ValueError("refinement_handle metadata has an invalid shape")
                metadata = json.loads(str(metadata_value.item()))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("refinement_handle is unreadable or corrupted") from exc
        if mask_logits.ndim != 3 or mask_logits.shape[0] != 1:
            raise ValueError("refinement_handle mask logits have an invalid shape")
        if not np.isfinite(mask_logits).all():
            raise ValueError("refinement_handle mask logits must be finite")
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ValueError("refinement_handle metadata version is unsupported")

        image = self._open_image(image_path)
        width, height = image.size
        expected_image = str(Path(image_path).resolve())
        if metadata.get("image_path") != expected_image:
            raise ValueError("refinement_handle belongs to a different image")
        if metadata.get("orig_img_w") != width or metadata.get("orig_img_h") != height:
            raise ValueError("refinement_handle image dimensions do not match")
        foreground_points = metadata.get("foreground_points", [])
        background_points = metadata.get("background_points", [])
        box = metadata.get("box")
        if foreground_points:
            self._validate_points(foreground_points, width, height, "foreground_points")
        if background_points:
            self._validate_points(background_points, width, height, "background_points")
        if box is not None:
            self._validate_box(box, width, height)
        if not foreground_points and not background_points and box is None:
            raise ValueError("refinement_handle contains no prompt history")
        return mask_logits, metadata

    def _persist_interactive_result(
        self,
        image_path: str,
        output_dir: str,
        mode: str,
        foreground_points: list[list[float]],
        background_points: list[list[float]],
        box: Optional[list[float]],
        source_handle: Optional[str],
        mask_input: Optional[np.ndarray],
        multimask_output: bool,
        verbose: bool,
    ) -> SegmentationResult:
        '''预测、保存交互结果并生成逐候选 refinement handles。'''

        prompt_data = {
            "foreground_points": foreground_points,
            "background_points": background_points,
            "box": box,
            "source_refinement_handle": source_handle,
        }
        artifact_stem = self._unique_artifact_stem(
            image_path,
            output_dir,
            mode,
            prompt_data,
        )
        if verbose:
            print(f"Running SAM3 interactive mode {mode!r} for {image_path!r}")
        masks, scores, low_res_masks, width, height = self._predict_interactive(
            image_path,
            foreground_points,
            background_points,
            box,
            mask_input,
            multimask_output,
        )
        handle_metadata = {
            "version": 1,
            "image_path": str(Path(image_path).resolve()),
            "orig_img_h": height,
            "orig_img_w": width,
            "foreground_points": foreground_points,
            "background_points": background_points,
            "box": box,
        }
        refinement_handles = self._save_refinement_handles(
            image_path,
            output_dir,
            artifact_stem,
            low_res_masks,
            handle_metadata,
        )
        data = {
            "orig_img_h": height,
            "orig_img_w": width,
            "pred_boxes": self._boxes_from_masks(masks, width, height),
            "pred_masks": self._encode_masks(masks),
            "pred_scores": scores.astype(float).tolist(),
            "prediction_mode": mode,
            "prompt_data": prompt_data,
            "refinement_handles": refinement_handles,
        }
        return self._persist_result(
            image_path,
            output_dir,
            artifact_stem,
            data,
        )

    def segment_instance_with_foreground_points(
        self,
        image_path: str,
        points: list[list[float]],
        output_dir: str,
        refinement_handle: Optional[str] = None,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据前景点创建实例或通过历史 logits 继续细化。'''

        if not isinstance(points, list) or not points:
            raise ValueError("points must be a non-empty array")
        if refinement_handle is None:
            foreground_points = points
            background_points: list[list[float]] = []
            box = None
            mask_input = None
            multimask_output = len(points) == 1
        else:
            mask_input, metadata = self._load_refinement_handle(
                image_path,
                output_dir,
                refinement_handle,
            )
            foreground_points = metadata.get("foreground_points", []) + points
            background_points = metadata.get("background_points", [])
            box = metadata.get("box")
            multimask_output = False
        return self._persist_interactive_result(
            image_path,
            output_dir,
            "instance_foreground_points",
            foreground_points,
            background_points,
            box,
            refinement_handle,
            mask_input,
            multimask_output,
            verbose,
        )

    def refine_instance_with_background_points(
        self,
        image_path: str,
        points: list[list[float]],
        refinement_handle: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据背景点和历史 logits 排除已有实例中的错误区域。'''

        if not isinstance(points, list) or not points:
            raise ValueError("points must be a non-empty array")
        mask_input, metadata = self._load_refinement_handle(
            image_path,
            output_dir,
            refinement_handle,
        )
        foreground_points = metadata.get("foreground_points", [])
        background_points = metadata.get("background_points", []) + points
        box = metadata.get("box")
        return self._persist_interactive_result(
            image_path,
            output_dir,
            "instance_background_points",
            foreground_points,
            background_points,
            box,
            refinement_handle,
            mask_input,
            False,
            verbose,
        )

    def segment_instance_with_box(
        self,
        image_path: str,
        box: list[float],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据像素定位框分割指定实例。'''

        return self._persist_interactive_result(
            image_path,
            output_dir,
            "instance_box",
            [],
            [],
            box,
            None,
            None,
            False,
            verbose,
        )

    def segment_instance_with_points_and_box(
        self,
        image_path: str,
        box: list[float],
        points: list[dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据带标签的像素点和定位框分割指定实例。'''

        image = self._open_image(image_path)
        width, height = image.size
        foreground_points, background_points = self._split_labeled_points(
            points,
            width,
            height,
        )
        validated_box = self._validate_box(box, width, height)
        return self._persist_interactive_result(
            image_path,
            output_dir,
            "instance_points_and_box",
            foreground_points,
            background_points,
            validated_box,
            None,
            None,
            False,
            verbose,
        )

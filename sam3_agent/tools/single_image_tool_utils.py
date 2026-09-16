'''单图分割工具共享的 schema 与结果组装逻辑。'''

from typing import Any, Optional

from .protocol import SegmentationResult, ToolContext, ToolResult


PIXEL_BOX_SCHEMA = {
    "type": "array",
    "items": {"type": "number", "minimum": 0},
    "minItems": 4,
    "maxItems": 4,
    "description": "Pixel-space box in [x1, y1, x2, y2] format.",
}

PIXEL_POINT_SCHEMA = {
    "type": "array",
    "items": {"type": "number", "minimum": 0},
    "minItems": 2,
    "maxItems": 2,
    "description": "Pixel-space point in [x, y] format.",
}

EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "box": PIXEL_BOX_SCHEMA,
        "label": {"type": "string", "enum": ["positive", "negative"]},
    },
    "required": ["box", "label"],
    "additionalProperties": False,
}

LABELED_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "point": PIXEL_POINT_SCHEMA,
        "label": {"type": "string", "enum": ["foreground", "background"]},
    },
    "required": ["point", "label"],
    "additionalProperties": False,
}


def require_non_empty_list(value: Any, field_name: str) -> list:
    '''校验字段为非空数组并返回原值。'''

    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    return value


def require_nullable_handle(value: Any, field_name: str) -> Optional[str]:
    '''校验 refinement handle 为非空字符串或空值。'''

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or null")
    return value.strip()


def finish_segmentation_tool(
    context: ToolContext,
    result: SegmentationResult,
    prompt_label: str,
    extra_content: Optional[dict[str, Any]] = None,
) -> ToolResult:
    '''更新共享上下文并构造统一的非终止工具结果。'''

    context.latest_output_json = result.json_path
    context.latest_text_prompt = prompt_label
    context.current_outputs = result.data
    mask_count = len(result.data.get("pred_masks", []))
    content: dict[str, Any] = {
        "status": "ok",
        "mask_count": mask_count,
        "output_json_path": result.json_path,
        "output_image_path": result.image_path,
    }
    handles = result.data.get("refinement_handles")
    if handles is not None:
        content["refinement_handles"] = handles
    if extra_content:
        content.update(extra_content)
    return ToolResult(
        content=content,
        image_path=result.image_path if mask_count else None,
    )

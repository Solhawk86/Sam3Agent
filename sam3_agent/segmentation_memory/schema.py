'''复合工具的严格 JSON schema。'''

from typing import Any

from ..tools.single_image_tool_utils import PIXEL_BOX_SCHEMA


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    '''构造所有字段必填且禁止额外字段的对象定义。'''

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def advance_schema(max_boxes: int) -> dict[str, Any]:
    '''按批次限制生成统一决策工具的参数定义。'''

    identifier = {"type": "string", "minLength": 1}
    reason = {"type": "string", "minLength": 1}
    review_item = object_schema({"mask_id": identifier, "reason": reason})
    identifiers = {"type": "array", "items": identifier, "minItems": 1}
    return object_schema(
        {
            "review": object_schema(
                {
                    "accept": {"type": "array", "items": review_item},
                    "reject": {"type": "array", "items": review_item},
                    "replace": {
                        "type": "array",
                        "items": object_schema(
                            {
                                "old_mask_ids": identifiers,
                                "new_mask_ids": identifiers,
                                "reason": reason,
                            }
                        ),
                    },
                }
            ),
            "text_prompt": {"type": ["string", "null"], "minLength": 1},
            "boxes": {
                "type": "array",
                "maxItems": max_boxes,
                "items": object_schema({"box": PIXEL_BOX_SCHEMA}),
            },
            "inspect_mask_ids": {"type": "array", "items": identifier, "maxItems": 4},
            "finish": {"type": "boolean"},
            "finish_reason": {
                "type": ["string", "null"],
                "enum": ["complete", "no_target", None],
            },
        }
    )

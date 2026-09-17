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

    identifier = {
        "type": "string",
        "minLength": 1,
        "description": "A non-empty mask ID present in the current memory summary.",
    }
    reason = {
        "type": "string",
        "minLength": 1,
        "description": "A non-empty reason for the mask review decision.",
    }
    review_item = object_schema({"mask_id": identifier, "reason": reason})
    review_item["description"] = (
        "One mask review item containing a current mask ID and a non-empty reason."
    )
    identifiers = {
        "type": "array",
        "items": identifier,
        "minItems": 1,
        "description": "A non-empty array of mask IDs.",
    }
    replacement = object_schema(
        {
            "old_mask_ids": identifiers,
            "new_mask_ids": identifiers,
            "reason": reason,
        }
    )
    replacement["description"] = (
        "Replace one or more accepted masks with one or more current masks; both "
        "ID arrays and the reason must be non-empty."
    )
    review = object_schema(
        {
            "accept": {
                "type": "array",
                "items": review_item,
                "description": "Masks to mark as accepted; may be empty.",
            },
            "reject": {
                "type": "array",
                "items": review_item,
                "description": "Masks to mark as rejected; may be empty.",
            },
            "replace": {
                "type": "array",
                "items": replacement,
                "description": "Accepted-mask replacement operations; may be empty.",
            },
        }
    )
    review["description"] = (
        "Status updates for masks visible in the current memory summary. A mask ID "
        "must not appear in conflicting or repeated review operations."
    )
    return object_schema(
        {
            "review": review,
            "text_prompt": {
                "type": ["string", "null"],
                "minLength": 1,
                "description": (
                    "One non-empty text phrase for segmentation, or null when no "
                    "text task is requested."
                ),
            },
            "boxes": {
                "type": "array",
                "maxItems": max_boxes,
                "items": object_schema({"box": PIXEL_BOX_SCHEMA}),
                "description": (
                    f"Zero to {max_boxes} instance boxes in original-image pixel "
                    "coordinates; each item must contain exactly one box."
                ),
            },
            "inspect_mask_ids": {
                "type": "array",
                "items": identifier,
                "maxItems": 4,
                "description": (
                    "Zero to four unique mask IDs from the current memory summary "
                    "for additional inspection views. All pending masks already "
                    "have automatic close-ups in the board; do not request IDs "
                    "already listed in inspection_mask_ids."
                ),
            },
            "finish": {
                "type": "boolean",
                "description": "Whether to terminate the agent after this decision.",
            },
            "finish_reason": {
                "type": ["string", "null"],
                "enum": ["complete", "no_target", None],
                "description": (
                    "Termination reason: complete or no_target when finish is true; "
                    "null when finish is false."
                ),
            },
        }
    )

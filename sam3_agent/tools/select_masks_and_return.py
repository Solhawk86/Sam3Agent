'''select_masks_and_return 原生 Agent 工具。'''

from ..viz import visualize
from .protocol import (
    BaseAgentTool,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)


class SelectMasksAndReturnTool(BaseAgentTool):
    '''从当前候选中选择最终 masks 并结束 Agent。'''

    name = "select_masks_and_return"
    description = (
        "Select one or more masks from the most recently rendered image as the "
        "complete final answer and end the grounding run."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "final_answer_masks": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 100},
                "minItems": 1,
                "maxItems": 100,
                "description": (
                    "Unique one-based mask numbers from the latest image. "
                    "Do not repeat an index."
                ),
            }
        },
        "required": ["final_answer_masks"],
        "additionalProperties": False,
    }
    requires_masks = True

    def execute(
        self,
        context: ToolContext,
        arguments: dict,
    ) -> ToolResult:
        '''校验编号并构造最终分割输出。'''

        require_exact_arguments(self.name, arguments, {"final_answer_masks"})
        selected = arguments["final_answer_masks"]
        if not isinstance(selected, list) or not selected:
            raise ValueError("final_answer_masks must be a non-empty array")
        if len(selected) > 100:
            raise ValueError("final_answer_masks cannot contain more than 100 items")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in selected):
            raise ValueError("final_answer_masks must contain only integers")
        if len(set(selected)) != len(selected):
            raise ValueError("final_answer_masks must not contain duplicates")
        if not context.current_outputs:
            raise ValueError("select_masks_and_return requires current outputs")

        available_count = len(context.current_outputs["pred_masks"])
        invalid = [item for item in selected if item < 1 or item > available_count]
        if invalid:
            raise ValueError(
                f"Mask indices {invalid} are outside the available range "
                f"1..{available_count}"
            )

        selected = sorted(selected)
        current_outputs = context.current_outputs
        final_outputs = {
            "original_image_path": current_outputs["original_image_path"],
            "orig_img_h": current_outputs["orig_img_h"],
            "orig_img_w": current_outputs["orig_img_w"],
            "pred_boxes": [current_outputs["pred_boxes"][i - 1] for i in selected],
            "pred_scores": [
                current_outputs["pred_scores"][i - 1] for i in selected
            ],
            "pred_masks": [current_outputs["pred_masks"][i - 1] for i in selected],
        }
        rendered = visualize(final_outputs)
        return ToolResult(
            content={"status": "ok", "selected_masks": selected},
            terminal=True,
            final_outputs=final_outputs,
            rendered_image=rendered,
        )

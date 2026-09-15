'''examine_each_mask 原生 Agent 工具。'''

import json
from pathlib import Path
from typing import Any

from ..viz import visualize
from .protocol import (
    BaseAgentTool,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)


class ExamineEachMaskTool(BaseAgentTool):
    '''逐个放大候选 mask，并让视觉模型过滤错误候选。'''

    name = "examine_each_mask"
    description = (
        "Individually render and inspect the currently available masks when they "
        "are small or overlapping, then keep only masks accepted by the vision model."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    requires_masks = True

    @staticmethod
    def _response_content(response: Any) -> str:
        '''读取逐 mask 检查请求返回的普通文本内容。'''

        if response is None:
            raise ValueError("Mask checking LLM returned no response")
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content:
            raise ValueError("Mask checking LLM returned no text content")
        return content

    def execute(
        self,
        context: ToolContext,
        arguments: dict,
    ) -> ToolResult:
        '''逐个检查当前 masks，保存过滤后的结果并更新上下文。'''

        require_exact_arguments(self.name, arguments, set())
        if not context.current_outputs or not context.latest_output_json:
            raise ValueError("examine_each_mask requires current segmentation outputs")

        current_outputs = context.current_outputs
        masks_to_keep: list[int] = []
        safe_prompt = context.latest_text_prompt.replace("/", "_")

        for index in range(len(current_outputs["pred_masks"])):
            image_with_mask, zoomed_image = visualize(current_outputs, index)
            image_with_mask_path = Path(context.sam_output_dir) / (
                f"{safe_prompt}_selected_mask_{index + 1}.png"
            )
            zoomed_image_path = Path(context.sam_output_dir) / (
                f"{safe_prompt}_zoom_in_mask_{index + 1}.png"
            )
            image_with_mask.save(image_with_mask_path)
            zoomed_image.save(zoomed_image_path)

            messages = [
                {"role": "system", "content": context.iterative_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The raw input image:"},
                        {"type": "image", "image": context.image_path},
                        {
                            "type": "text",
                            "text": (
                                "The initial user input query is: "
                                f"{context.initial_text_prompt!r}"
                            ),
                        },
                        {
                            "type": "text",
                            "text": "Image with the predicted mask rendered on it:",
                        },
                        {"type": "image", "image": str(image_with_mask_path)},
                        {
                            "type": "text",
                            "text": "Image with the zoomed-in mask:",
                        },
                        {"type": "image", "image": str(zoomed_image_path)},
                    ],
                },
            ]
            context.mask_check_count += 1
            response = context.send_generate_request(messages)
            context.save_llm_output(
                response,
                (
                    f"raw_mask_check_{context.mask_check_count:03d}_"
                    f"mask_{index + 1:03d}.txt"
                ),
            )
            checking_text = self._response_content(response)
            if "<verdict>" not in checking_text or "</verdict>" not in checking_text:
                raise ValueError(
                    "Mask checking response must contain a <verdict> element: "
                    f"{checking_text}"
                )
            verdict = (
                checking_text.split("<verdict>", 1)[1]
                .split("</verdict>", 1)[0]
                .strip()
            )
            if verdict == "Accept":
                masks_to_keep.append(index)
            elif verdict != "Reject":
                raise ValueError(
                    f"Unexpected mask verdict {verdict!r}; expected Accept or Reject"
                )

        updated_outputs = {
            "original_image_path": current_outputs["original_image_path"],
            "orig_img_h": current_outputs["orig_img_h"],
            "orig_img_w": current_outputs["orig_img_w"],
            "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
            "pred_scores": [
                current_outputs["pred_scores"][i] for i in masks_to_keep
            ],
            "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
        }

        selected_suffix = (
            "_".join(str(index + 1) for index in masks_to_keep)
            if masks_to_keep
            else "none"
        )
        rendered_path = Path(context.sam_output_dir) / (
            f"{safe_prompt}_selected_masks_{selected_suffix}.png"
        )
        rendered = visualize(updated_outputs)
        rendered.save(rendered_path)
        updated_outputs["output_image_path"] = str(rendered_path)

        source_path = Path(context.latest_output_json)
        updated_json_path = source_path.with_name(
            f"{source_path.stem.split('_masks_', 1)[0]}_masks_{selected_suffix}.json"
        )
        with updated_json_path.open("w", encoding="utf-8") as handle:
            json.dump(updated_outputs, handle, indent=4)

        context.latest_output_json = str(updated_json_path)
        context.current_outputs = updated_outputs
        mask_count = len(masks_to_keep)
        if mask_count:
            message = (
                f"Accepted {mask_count} masks. They have been renumbered in the "
                "attached image; compare them with the original query before choosing."
            )
            image_path = str(rendered_path)
        else:
            message = (
                "All masks were rejected. Call segment_phrase again with a different "
                "simple noun phrase."
            )
            image_path = None

        return ToolResult(
            content={
                "status": "ok",
                "accepted_mask_count": mask_count,
                "message": message,
            },
            image_path=image_path,
        )

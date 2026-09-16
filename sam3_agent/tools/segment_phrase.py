'''文本概念分割的原生 Agent 工具。'''

from .protocol import (
    BaseAgentTool,
    SegmentationTool,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)
from .single_image_backend import Sam3Tool


class SegmentPhraseTool(BaseAgentTool):
    '''把文本概念分割器封装成原生 function-call 工具。'''

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


__all__ = ["Sam3Tool", "SegmentPhraseTool"]

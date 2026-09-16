'''文本与视觉样例框联合概念分割工具。'''

from .protocol import (
    BaseAgentTool,
    SingleImageSegmentationBackend,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)
from .single_image_tool_utils import (
    EXAMPLE_SCHEMA,
    finish_segmentation_tool,
    require_non_empty_list,
)


class SegmentPhraseWithVisualExamplesTool(BaseAgentTool):
    '''联合文本和正负样例框找出全图同类实例。'''

    name = "segment_phrase_with_visual_examples"
    description = (
        "Segment every instance matching a text concept refined by positive or "
        "negative visual example boxes. Box coordinates are pixels."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text_prompt": {"type": "string", "minLength": 1},
            "examples": {
                "type": "array",
                "items": EXAMPLE_SCHEMA,
                "minItems": 1,
                "maxItems": 100,
            },
        },
        "required": ["text_prompt", "examples"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验文本和视觉样例并执行联合概念分割。'''

        require_exact_arguments(
            self.name,
            arguments,
            {"text_prompt", "examples"},
        )
        text_prompt = arguments["text_prompt"]
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        text_prompt = text_prompt.strip()
        examples = require_non_empty_list(arguments["examples"], "examples")
        result = self.segmentation_backend.segment_phrase_with_visual_examples(
            image_path=context.image_path,
            text_prompt=text_prompt,
            examples=examples,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            text_prompt,
            {"text_prompt": text_prompt, "example_count": len(examples)},
        )

'''正负视觉样例框概念分割工具。'''

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


class SegmentVisualExamplesTool(BaseAgentTool):
    '''使用正负视觉样例框找出全图同类实例。'''

    name = "segment_visual_examples"
    description = (
        "Segment every instance matching positive visual example boxes while "
        "using negative boxes to exclude confusing concepts. Coordinates are pixels."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": EXAMPLE_SCHEMA,
                "minItems": 1,
                "maxItems": 100,
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验视觉样例并执行概念分割。'''

        require_exact_arguments(self.name, arguments, {"examples"})
        examples = require_non_empty_list(arguments["examples"], "examples")
        result = self.segmentation_backend.segment_visual_examples(
            image_path=context.image_path,
            examples=examples,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            "visual_examples",
            {"example_count": len(examples)},
        )

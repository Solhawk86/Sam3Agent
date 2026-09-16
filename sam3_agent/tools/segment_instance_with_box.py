'''定位框交互式实例分割工具。'''

from .protocol import (
    BaseAgentTool,
    SingleImageSegmentationBackend,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)
from .single_image_tool_utils import (
    PIXEL_BOX_SCHEMA,
    finish_segmentation_tool,
    require_non_empty_list,
)


class SegmentInstanceWithBoxTool(BaseAgentTool):
    '''使用一个像素定位框分割指定实例。'''

    name = "segment_instance_with_box"
    description = (
        "Segment one specific object from one pixel-space [x1, y1, x2, y2] box."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"box": PIXEL_BOX_SCHEMA},
        "required": ["box"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验定位框并执行实例分割。'''

        require_exact_arguments(self.name, arguments, {"box"})
        box = require_non_empty_list(arguments["box"], "box")
        result = self.segmentation_backend.segment_instance_with_box(
            image_path=context.image_path,
            box=box,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            "instance_box",
        )

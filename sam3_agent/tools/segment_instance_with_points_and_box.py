'''点与定位框联合交互式实例分割工具。'''

from .protocol import (
    BaseAgentTool,
    SingleImageSegmentationBackend,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)
from .single_image_tool_utils import (
    LABELED_POINT_SCHEMA,
    PIXEL_BOX_SCHEMA,
    finish_segmentation_tool,
    require_non_empty_list,
)


class SegmentInstanceWithPointsAndBoxTool(BaseAgentTool):
    '''联合带标签的点和定位框分割指定实例。'''

    name = "segment_instance_with_points_and_box"
    description = (
        "Segment one specific object from a pixel-space box plus foreground or "
        "background pixel points."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "box": PIXEL_BOX_SCHEMA,
            "points": {
                "type": "array",
                "items": LABELED_POINT_SCHEMA,
                "minItems": 1,
                "maxItems": 100,
            },
        },
        "required": ["box", "points"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验点和定位框并执行联合实例分割。'''

        require_exact_arguments(self.name, arguments, {"box", "points"})
        box = require_non_empty_list(arguments["box"], "box")
        points = require_non_empty_list(arguments["points"], "points")
        result = self.segmentation_backend.segment_instance_with_points_and_box(
            image_path=context.image_path,
            box=box,
            points=points,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            "instance_points_and_box",
            {"point_count": len(points)},
        )

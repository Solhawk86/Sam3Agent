'''前景点交互式实例分割工具。'''

from .protocol import (
    BaseAgentTool,
    SingleImageSegmentationBackend,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)
from .single_image_tool_utils import (
    PIXEL_POINT_SCHEMA,
    finish_segmentation_tool,
    require_non_empty_list,
    require_nullable_handle,
)


class SegmentInstanceWithForegroundPointsTool(BaseAgentTool):
    '''使用前景点创建实例 mask 或继续细化已有实例。'''

    name = "segment_instance_with_foreground_points"
    description = (
        "Segment one specific object from foreground pixel points. Pass null as "
        "refinement_handle for a new object, or a returned handle to add points."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": PIXEL_POINT_SCHEMA,
                "minItems": 1,
                "maxItems": 100,
            },
            "refinement_handle": {"type": ["string", "null"]},
        },
        "required": ["points", "refinement_handle"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验前景点和句柄并执行实例分割。'''

        require_exact_arguments(
            self.name,
            arguments,
            {"points", "refinement_handle"},
        )
        points = require_non_empty_list(arguments["points"], "points")
        refinement_handle = require_nullable_handle(
            arguments["refinement_handle"],
            "refinement_handle",
        )
        result = self.segmentation_backend.segment_instance_with_foreground_points(
            image_path=context.image_path,
            points=points,
            refinement_handle=refinement_handle,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            "instance_foreground_points",
            {"added_foreground_point_count": len(points)},
        )

'''背景点交互式实例细化工具。'''

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


class RefineInstanceWithBackgroundPointsTool(BaseAgentTool):
    '''在已有实例上追加背景点以排除错误区域。'''

    name = "refine_instance_with_background_points"
    description = (
        "Refine one previously predicted instance by adding background pixel "
        "points. A refinement handle returned by an interactive tool is required."
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
            "refinement_handle": {"type": "string", "minLength": 1},
        },
        "required": ["points", "refinement_handle"],
        "additionalProperties": False,
    }

    def __init__(self, segmentation_backend: SingleImageSegmentationBackend):
        '''注入支持全部单图分割模式的 SAM3 后端。'''

        self.segmentation_backend = segmentation_backend

    def execute(self, context: ToolContext, arguments: dict) -> ToolResult:
        '''校验背景点和历史句柄并执行实例细化。'''

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
        if refinement_handle is None:
            raise ValueError("refinement_handle must be a non-empty string")
        result = self.segmentation_backend.refine_instance_with_background_points(
            image_path=context.image_path,
            points=points,
            refinement_handle=refinement_handle,
            output_dir=context.sam_output_dir,
            verbose=context.verbose,
        )
        return finish_segmentation_tool(
            context,
            result,
            "instance_background_points",
            {"added_background_point_count": len(points)},
        )

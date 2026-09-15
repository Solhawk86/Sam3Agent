'''report_no_mask 原生 Agent 工具。'''

from PIL import Image

from .protocol import (
    BaseAgentTool,
    ToolContext,
    ToolResult,
    require_exact_arguments,
)


class ReportNoMaskTool(BaseAgentTool):
    '''在图片中不存在匹配目标时返回空分割结果。'''

    name = "report_no_mask"
    description = (
        "End the grounding run with an empty result only when the raw image has no "
        "object that can match the original query."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def execute(
        self,
        context: ToolContext,
        arguments: dict,
    ) -> ToolResult:
        '''读取原图尺寸并构造空 masks 终止结果。'''

        require_exact_arguments(self.name, arguments, set())
        with Image.open(context.image_path) as image:
            width, height = image.size
            rendered = image.copy()
        final_outputs = {
            "original_image_path": context.image_path,
            "orig_img_h": height,
            "orig_img_w": width,
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
        }
        return ToolResult(
            content={"status": "ok", "message": "No matching masks found."},
            terminal=True,
            final_outputs=final_outputs,
            rendered_image=rendered,
        )

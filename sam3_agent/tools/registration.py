'''当前 grounding Agent 使用的工具注册组合。'''

from .examine_each_mask import ExamineEachMaskTool
from .protocol import SegmentationTool, ToolRegistry
from .report_no_mask import ReportNoMaskTool
from .segment_phrase import SegmentPhraseTool
from .select_masks_and_return import SelectMasksAndReturnTool


def build_agent_tool_registry(
    segmentation_tool: SegmentationTool,
) -> ToolRegistry:
    '''使用注入的分割后端创建当前四工具注册表。'''

    return ToolRegistry(
        [
            SegmentPhraseTool(segmentation_tool),
            ExamineEachMaskTool(),
            SelectMasksAndReturnTool(),
            ReportNoMaskTool(),
        ]
    )


__all__ = ["build_agent_tool_registry"]

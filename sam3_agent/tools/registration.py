'''批量记忆 Agent 的唯一工具注册入口。'''

from .advance_segmentation import AdvanceSegmentationTool
from .protocol import SingleImageSegmentationBackend, ToolRegistry


def build_agent_tool_registry(
    segmentation_tool: SingleImageSegmentationBackend,
    max_box_tasks_per_round: int = 4,
) -> ToolRegistry:
    '''注册复合决策工具，独立分割适配器仅供内部复用。'''

    return ToolRegistry(
        [
            AdvanceSegmentationTool(segmentation_tool, max_box_tasks_per_round),
        ]
    )

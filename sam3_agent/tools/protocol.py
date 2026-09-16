'''Agent 工具的公共协议、运行上下文与注册表。'''

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

from PIL import Image


@dataclass(frozen=True)
class SegmentationResult:
    '''描述分割后落盘的 JSON、渲染图和结构化数据。'''

    json_path: str
    image_path: str
    data: Dict[str, Any]


class SegmentationTool(Protocol):
    '''定义底层图文分割器需要实现的接口。'''

    def segment_phrase(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据短语返回 masks、boxes、scores 和渲染图。'''


class SingleImageSegmentationBackend(SegmentationTool, Protocol):
    '''定义七种单图分割工具共用的 SAM3 后端接口。'''

    def segment_visual_examples(
        self,
        image_path: str,
        examples: list[Dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据正负视觉样例框执行概念分割。'''

    def segment_phrase_with_visual_examples(
        self,
        image_path: str,
        text_prompt: str,
        examples: list[Dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据文本和正负视觉样例框执行概念分割。'''

    def segment_instance_with_foreground_points(
        self,
        image_path: str,
        points: list[list[float]],
        output_dir: str,
        refinement_handle: Optional[str] = None,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据前景点创建或细化一个实例。'''

    def refine_instance_with_background_points(
        self,
        image_path: str,
        points: list[list[float]],
        refinement_handle: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据背景点和历史 logits 细化一个实例。'''

    def segment_instance_with_box(
        self,
        image_path: str,
        box: list[float],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据定位框分割一个实例。'''

    def segment_instance_with_points_and_box(
        self,
        image_path: str,
        box: list[float],
        points: list[Dict[str, Any]],
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        '''根据带标签的点和定位框分割一个实例。'''


@dataclass
class ToolContext:
    '''保存四个工具共享的单图 Agent 运行状态。'''

    image_path: str
    initial_text_prompt: str
    sam_output_dir: str
    iterative_system_prompt: str
    send_generate_request: Callable[..., Any]
    save_llm_output: Callable[[Any, str], None]
    verbose: bool = False
    latest_output_json: str = ""
    latest_text_prompt: str = ""
    latest_segment_call_id: Optional[str] = None
    current_outputs: Optional[Dict[str, Any]] = None
    used_text_prompts: set[str] = field(default_factory=set)
    mask_check_count: int = 0

    @property
    def has_masks(self) -> bool:
        '''返回当前上下文中是否存在可供选择的 mask。'''

        return bool(self.current_outputs and self.current_outputs.get("pred_masks"))


@dataclass(frozen=True)
class ToolResult:
    '''统一描述一次工具执行产生的模型反馈和终止结果。'''

    content: Dict[str, Any]
    image_path: Optional[str] = None
    terminal: bool = False
    success: bool = True
    final_outputs: Optional[Dict[str, Any]] = None
    rendered_image: Optional[Image.Image] = None

    def as_tool_content(self) -> str:
        '''把结构化工具结果序列化成原生 tool 消息内容。'''

        import json

        return json.dumps(self.content, ensure_ascii=False)


class AgentTool(Protocol):
    '''定义可注册、可生成 schema、可统一执行的 Agent 工具。'''

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_masks: bool

    @property
    def definition(self) -> Dict[str, Any]:
        '''返回 OpenAI-compatible function tool 定义。'''

    def execute(
        self,
        context: ToolContext,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        '''校验参数并在给定上下文中执行工具。'''


class BaseAgentTool:
    '''为具体工具提供统一的原生 function schema。'''

    name = ""
    description = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    requires_masks = False

    @property
    def definition(self) -> Dict[str, Any]:
        '''返回 OpenAI-compatible function tool 定义。'''

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
                "strict": True,
            },
        }


class ToolRegistry:
    '''注册工具并根据当前状态导出 schema 或分发调用。'''

    def __init__(self, tools: Iterable[AgentTool]):
        '''创建注册表并拒绝重名工具。'''

        self._tools: Dict[str, AgentTool] = {}
        for tool in tools:
            if not tool.name:
                raise ValueError("Tool name must not be empty")
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def available_tools(self, context: ToolContext) -> list[AgentTool]:
        '''返回当前状态下允许模型调用的工具。'''

        return [
            tool
            for tool in self._tools.values()
            if not tool.requires_masks or context.has_masks
        ]

    def definitions(self, context: ToolContext) -> list[Dict[str, Any]]:
        '''导出当前可用工具的 OpenAI-compatible schemas。'''

        return [tool.definition for tool in self.available_tools(context)]

    def execute(
        self,
        name: str,
        context: ToolContext,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        '''按名称查找工具、检查状态并执行。'''

        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool call: {name}")
        if tool.requires_masks and not context.has_masks:
            raise ValueError(f"Tool {name!r} requires at least one available mask")
        return tool.execute(context, arguments)


def require_exact_arguments(
    tool_name: str,
    arguments: Dict[str, Any],
    expected_keys: set[str],
) -> None:
    '''校验工具参数对象只包含预期字段。'''

    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {tool_name!r} must be a JSON object")
    actual_keys = set(arguments)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Arguments for {tool_name!r} must contain exactly "
            f"{sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

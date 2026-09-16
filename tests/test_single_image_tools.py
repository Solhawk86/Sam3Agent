from pathlib import Path

import pytest

from sam3_agent.tools import (
    RefineInstanceWithBackgroundPointsTool,
    SegmentationResult,
    SegmentInstanceWithBoxTool,
    SegmentInstanceWithForegroundPointsTool,
    SegmentInstanceWithPointsAndBoxTool,
    SegmentPhraseTool,
    SegmentPhraseWithVisualExamplesTool,
    SegmentVisualExamplesTool,
    ToolContext,
)


class FakeSingleImageBackend:
    '''记录单图工具转发参数并返回固定结果。'''

    def __init__(self):
        '''创建空调用记录。'''

        self.calls = []

    def _result(self, image_path, output_dir, mode):
        '''构造无需实际文件的固定分割结果。'''

        data = {
            "original_image_path": image_path,
            "output_image_path": str(Path(output_dir) / f"{mode}.png"),
            "orig_img_h": 10,
            "orig_img_w": 12,
            "pred_boxes": [[0.1, 0.1, 0.5, 0.5]],
            "pred_masks": ["encoded"],
            "pred_scores": [0.9],
            "refinement_handles": [
                {"mask_number": 1, "refinement_handle": "candidate.npz"}
            ],
        }
        return SegmentationResult(
            str(Path(output_dir) / f"{mode}.json"),
            data["output_image_path"],
            data,
        )

    def segment_phrase(self, image_path, text_prompt, output_dir, verbose=False):
        '''记录文本概念分割调用。'''

        self.calls.append(("phrase", text_prompt))
        return self._result(image_path, output_dir, "phrase")

    def segment_visual_examples(
        self, image_path, examples, output_dir, verbose=False
    ):
        '''记录视觉样例概念分割调用。'''

        self.calls.append(("visual", examples))
        return self._result(image_path, output_dir, "visual")

    def segment_phrase_with_visual_examples(
        self, image_path, text_prompt, examples, output_dir, verbose=False
    ):
        '''记录文本与视觉样例联合调用。'''

        self.calls.append(("text_visual", text_prompt, examples))
        return self._result(image_path, output_dir, "text_visual")

    def segment_instance_with_foreground_points(
        self,
        image_path,
        points,
        output_dir,
        refinement_handle=None,
        verbose=False,
    ):
        '''记录前景点实例分割调用。'''

        self.calls.append(("foreground", points, refinement_handle))
        return self._result(image_path, output_dir, "foreground")

    def refine_instance_with_background_points(
        self,
        image_path,
        points,
        refinement_handle,
        output_dir,
        verbose=False,
    ):
        '''记录背景点实例细化调用。'''

        self.calls.append(("background", points, refinement_handle))
        return self._result(image_path, output_dir, "background")

    def segment_instance_with_box(
        self, image_path, box, output_dir, verbose=False
    ):
        '''记录定位框实例分割调用。'''

        self.calls.append(("box", box))
        return self._result(image_path, output_dir, "box")

    def segment_instance_with_points_and_box(
        self, image_path, box, points, output_dir, verbose=False
    ):
        '''记录点框联合实例分割调用。'''

        self.calls.append(("points_box", box, points))
        return self._result(image_path, output_dir, "points_box")


def make_context(tmp_path: Path) -> ToolContext:
    '''构造不依赖 LLM 的工具运行上下文。'''

    return ToolContext(
        image_path=str(tmp_path / "input.png"),
        initial_text_prompt="object",
        sam_output_dir=str(tmp_path / "sam"),
        iterative_system_prompt="check",
        send_generate_request=lambda messages, **kwargs: None,
        save_llm_output=lambda response, filename: None,
    )


def make_seven_tools(backend: FakeSingleImageBackend):
    '''构造计划定义的七个单图分割工具。'''

    return [
        SegmentPhraseTool(backend),
        SegmentVisualExamplesTool(backend),
        SegmentPhraseWithVisualExamplesTool(backend),
        SegmentInstanceWithForegroundPointsTool(backend),
        RefineInstanceWithBackgroundPointsTool(backend),
        SegmentInstanceWithBoxTool(backend),
        SegmentInstanceWithPointsAndBoxTool(backend),
    ]


def test_seven_tools_export_strict_independent_schemas():
    '''验证七个工具名称、独立模块和严格根 schema。'''

    tools = make_seven_tools(FakeSingleImageBackend())
    assert [tool.name for tool in tools] == [
        "segment_phrase",
        "segment_visual_examples",
        "segment_phrase_with_visual_examples",
        "segment_instance_with_foreground_points",
        "refine_instance_with_background_points",
        "segment_instance_with_box",
        "segment_instance_with_points_and_box",
    ]
    assert len({tool.__class__.__module__ for tool in tools}) == 7
    for tool in tools:
        definition = tool.definition["function"]
        assert definition["strict"] is True
        assert definition["parameters"]["type"] == "object"
        assert definition["parameters"]["additionalProperties"] is False
        assert set(definition["parameters"]["required"]) == set(
            definition["parameters"]["properties"]
        )


def test_concept_tools_forward_native_arguments(tmp_path):
    '''验证视觉样例和文本联合工具的参数转发及上下文更新。'''

    backend = FakeSingleImageBackend()
    context = make_context(tmp_path)
    examples = [
        {"box": [1, 2, 5, 8], "label": "positive"},
        {"box": [6, 2, 10, 8], "label": "negative"},
    ]
    visual_result = SegmentVisualExamplesTool(backend).execute(
        context,
        {"examples": examples},
    )
    combined_result = SegmentPhraseWithVisualExamplesTool(backend).execute(
        context,
        {"text_prompt": " crab ", "examples": examples},
    )

    assert backend.calls == [
        ("visual", examples),
        ("text_visual", "crab", examples),
    ]
    assert visual_result.content["example_count"] == 2
    assert combined_result.content["text_prompt"] == "crab"
    assert context.latest_text_prompt == "crab"
    assert context.current_outputs is not None


def test_interactive_tools_forward_points_boxes_and_handles(tmp_path):
    '''验证四种交互工具把点、框和句柄原样交给后端。'''

    backend = FakeSingleImageBackend()
    context = make_context(tmp_path)
    foreground = SegmentInstanceWithForegroundPointsTool(backend).execute(
        context,
        {"points": [[2, 3]], "refinement_handle": None},
    )
    RefineInstanceWithBackgroundPointsTool(backend).execute(
        context,
        {"points": [[4, 5]], "refinement_handle": "one.npz"},
    )
    SegmentInstanceWithBoxTool(backend).execute(
        context,
        {"box": [1, 2, 8, 9]},
    )
    labeled_points = [
        {"point": [2, 3], "label": "foreground"},
        {"point": [4, 5], "label": "background"},
    ]
    SegmentInstanceWithPointsAndBoxTool(backend).execute(
        context,
        {"box": [1, 2, 8, 9], "points": labeled_points},
    )

    assert backend.calls == [
        ("foreground", [[2, 3]], None),
        ("background", [[4, 5]], "one.npz"),
        ("box", [1, 2, 8, 9]),
        ("points_box", [1, 2, 8, 9], labeled_points),
    ]
    assert foreground.content["refinement_handles"][0]["mask_number"] == 1
    assert foreground.image_path is not None


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (SegmentVisualExamplesTool, {"examples": [], "extra": 1}),
        (
            SegmentPhraseWithVisualExamplesTool,
            {"text_prompt": "", "examples": []},
        ),
        (
            SegmentInstanceWithForegroundPointsTool,
            {"points": [], "refinement_handle": None},
        ),
        (
            RefineInstanceWithBackgroundPointsTool,
            {"points": [[1, 1]], "refinement_handle": None},
        ),
        (SegmentInstanceWithBoxTool, {"box": []}),
        (SegmentInstanceWithPointsAndBoxTool, {"box": [0, 0, 1, 1], "points": []}),
    ],
)
def test_tools_reject_invalid_top_level_arguments(tmp_path, tool, arguments):
    '''验证工具在调用后端前拒绝缺失、额外或空参数。'''

    backend = FakeSingleImageBackend()
    with pytest.raises(ValueError):
        tool(backend).execute(make_context(tmp_path), arguments)
    assert backend.calls == []

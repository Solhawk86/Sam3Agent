from pathlib import Path

from sam3_agent.tools import (
    ExamineEachMaskTool,
    ReportNoMaskTool,
    SegmentPhraseTool,
    SelectMasksAndReturnTool,
    ToolContext,
    ToolRegistry,
)


class StubSegmentationTool:
    def segment_phrase(self, image_path, text_prompt, output_dir, verbose=False):
        raise AssertionError("not called")


def make_context(tmp_path: Path):
    return ToolContext(
        image_path=str(tmp_path / "input.png"),
        initial_text_prompt="object",
        sam_output_dir=str(tmp_path),
        iterative_system_prompt="check",
        send_generate_request=lambda messages, **kwargs: None,
        save_llm_output=lambda response, filename: None,
    )


def test_registry_exports_state_dependent_native_schemas(tmp_path):
    registry = ToolRegistry(
        [
            SegmentPhraseTool(StubSegmentationTool()),
            ExamineEachMaskTool(),
            SelectMasksAndReturnTool(),
            ReportNoMaskTool(),
        ]
    )
    context = make_context(tmp_path)

    initial = registry.definitions(context)
    assert [item["function"]["name"] for item in initial] == [
        "segment_phrase",
        "report_no_mask",
    ]
    assert all(item["type"] == "function" for item in initial)
    assert all(item["function"]["strict"] is True for item in initial)
    assert all(
        item["function"]["parameters"]["additionalProperties"] is False
        for item in initial
    )

    context.current_outputs = {"pred_masks": ["encoded"]}
    available = registry.definitions(context)
    assert {item["function"]["name"] for item in available} == {
        "segment_phrase",
        "examine_each_mask",
        "select_masks_and_return",
        "report_no_mask",
    }


def test_tool_classes_come_from_independent_modules():
    assert SegmentPhraseTool.__module__.endswith(".segment_phrase")
    assert ExamineEachMaskTool.__module__.endswith(".examine_each_mask")
    assert SelectMasksAndReturnTool.__module__.endswith(".select_masks_and_return")
    assert ReportNoMaskTool.__module__.endswith(".report_no_mask")

from pathlib import Path

from sam3_agent.cli import build_parser
from sam3_agent.tools import Sam3Tool, SegmentPhraseTool
from sam3_agent.tools.sam3_tool import Sam3Tool as LegacyImportSam3Tool


def test_prompt_files_are_packaged():
    prompt_dir = Path(__file__).parents[1] / "sam3_agent" / "system_prompts"
    assert (prompt_dir / "system_prompt.txt").exists()
    prompt = (prompt_dir / "system_prompt.txt").read_text()
    assert "advance_segmentation" in prompt
    assert "<tool>" not in prompt
    assert not (prompt_dir / "system_prompt_iterative_checking.txt").exists()


def test_sam3_tool_is_lazy():
    tool = Sam3Tool(device="cpu")
    assert tool._processor is None
    assert tool.enable_inst_interactivity is False


def test_sam3_tool_can_enable_interactive_instance_head():
    '''验证交互实例头必须由调用方显式启用。'''

    tool = Sam3Tool(device="cpu", enable_inst_interactivity=True)
    assert tool._processor is None
    assert tool.enable_inst_interactivity is True


def test_segment_phrase_tool_wraps_sam3_backend():
    backend = Sam3Tool(device="cpu")
    tool = SegmentPhraseTool(backend)
    assert tool.segmentation_tool is backend


def test_legacy_sam3_import_path_is_preserved():
    assert LegacyImportSam3Tool is Sam3Tool


def test_cli_requires_image_and_prompt():
    args = build_parser().parse_args(["--image", "input.jpg", "--prompt", "object"])
    assert args.image == "input.jpg"
    assert args.prompt == "object"

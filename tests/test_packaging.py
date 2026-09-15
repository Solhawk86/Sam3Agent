from pathlib import Path

from sam3_agent.cli import build_parser
from sam3_agent.tools import Sam3Tool


def test_prompt_files_are_packaged():
    prompt_dir = Path(__file__).parents[1] / "sam3_agent" / "system_prompts"
    assert (prompt_dir / "system_prompt.txt").exists()
    assert (prompt_dir / "system_prompt_iterative_checking.txt").exists()


def test_sam3_tool_is_lazy():
    tool = Sam3Tool(device="cpu")
    assert tool._processor is None


def test_cli_requires_image_and_prompt():
    args = build_parser().parse_args(["--image", "input.jpg", "--prompt", "object"])
    assert args.image == "input.jpg"
    assert args.prompt == "object"

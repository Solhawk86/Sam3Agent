import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from sam3_agent.inference import run_single_image_inference
from sam3_agent.tools.protocol import SegmentationResult


def make_fixture_image(path: Path):
    Image.new("RGB", (12, 10), (240, 240, 240)).save(path)


def make_segmentation_result(image_path: Path, output_dir: Path):
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    result_dir = output_dir / "sam" / image_path.stem
    result_dir.mkdir(parents=True)
    rendered_path = result_dir / "object.png"
    Image.open(image_path).save(rendered_path)
    data = {
        "original_image_path": str(image_path),
        "output_image_path": str(rendered_path),
        "orig_img_h": 10,
        "orig_img_w": 12,
        "pred_boxes": [[0.25, 0.2, 0.5, 0.6]],
        "pred_masks": [encoded],
        "pred_scores": [0.9],
    }
    json_path = result_dir / "object.json"
    json_path.write_text(json.dumps(data))
    return SegmentationResult(str(json_path), str(rendered_path), data)


class FakeSegmentationTool:
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = output_dir
        self.calls = []

    def segment_phrase(self, image_path, text_prompt, output_dir, verbose=False):
        self.calls.append((image_path, text_prompt, output_dir))
        return make_segmentation_result(self.image_path, self.output_dir)


def test_segment_and_select_flow(tmp_path):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)
    responses = iter(
        [
            '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"object"}}</tool>',
            '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[1]}}</tool>',
        ]
    )

    result = run_single_image_inference(
        image_path=str(image_path),
        text_prompt="the object",
        llm_config={"name": "fake"},
        send_generate_request=lambda messages: next(responses),
        segmentation_tool=tool,
        output_dir=str(tmp_path / "outputs"),
        final_mask_output_dir=str(tmp_path / "final_masks"),
        verbose=False,
    )

    assert result["status"] == "success"
    assert len(tool.calls) == 1
    assert Path(result["output_json_path"]).exists()
    assert Path(result["output_image_path"]).exists()
    assert Path(result["final_mask_path"]).exists()


def test_report_no_mask_does_not_call_segmentation_tool(tmp_path):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)

    result = run_single_image_inference(
        image_path=str(image_path),
        text_prompt="a unicorn",
        llm_config={"name": "fake"},
        send_generate_request=lambda messages: (
            '<tool>{"name":"report_no_mask","parameters":{}}</tool>'
        ),
        segmentation_tool=tool,
        output_dir=str(tmp_path / "outputs"),
    )

    assert result["status"] == "success"
    assert tool.calls == []
    output = json.loads(Path(result["output_json_path"]).read_text())
    assert output["pred_masks"] == []


def test_duplicate_prompt_is_not_sent_to_segmentation_tool(tmp_path):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)
    responses = iter(
        [
            '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"object"}}</tool>',
            '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"object"}}</tool>',
            '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[1]}}</tool>',
        ]
    )

    run_single_image_inference(
        image_path=str(image_path),
        text_prompt="the object",
        llm_config={"name": "fake"},
        send_generate_request=lambda messages: next(responses),
        segmentation_tool=tool,
        output_dir=str(tmp_path / "outputs"),
    )

    assert [call[1] for call in tool.calls] == ["object"]


def test_examine_each_mask_accepts_and_returns_selected_mask(tmp_path):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)
    responses = iter(
        [
            '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"object"}}</tool>',
            '<tool>{"name":"examine_each_mask","parameters":{}}</tool>',
            "<think>the mask matches</think><verdict>Accept</verdict>",
            '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[1]}}</tool>',
        ]
    )

    result = run_single_image_inference(
        image_path=str(image_path),
        text_prompt="the object",
        llm_config={"name": "fake"},
        send_generate_request=lambda messages: next(responses),
        segmentation_tool=tool,
        output_dir=str(tmp_path / "outputs"),
    )

    assert result["status"] == "success"


def test_generation_limit_is_enforced(tmp_path):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)

    try:
        run_single_image_inference(
            image_path=str(image_path),
            text_prompt="the object",
            llm_config={"name": "fake"},
            send_generate_request=lambda messages: (
                '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"object"}}</tool>'
            ),
            segmentation_tool=tool,
            output_dir=str(tmp_path / "outputs"),
            max_generations=0,
        )
    except ValueError as exc:
        assert "maximum number" in str(exc)
    else:
        raise AssertionError("Expected max_generations to stop the agent")

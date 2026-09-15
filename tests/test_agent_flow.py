import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import pytest
from PIL import Image

from sam3_agent.agent_core import count_images
from sam3_agent.inference import run_single_image_inference
from sam3_agent.llm_client import FunctionToolCall, LLMResponse
from sam3_agent.tools.protocol import SegmentationResult


def make_fixture_image(path: Path):
    Image.new("RGB", (12, 10), (240, 240, 240)).save(path)


def make_segmentation_result(image_path: Path, output_dir: Path):
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    result_dir = output_dir / "sam" / image_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
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


def native_tool_response(name, arguments, call_id="call_1", content="analysis"):
    raw_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return LLMResponse(
        content=content,
        tool_calls=(FunctionToolCall(call_id, name, raw_arguments),),
    )


class FakeSegmentationTool:
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = output_dir
        self.calls = []

    def segment_phrase(self, image_path, text_prompt, output_dir, verbose=False):
        self.calls.append((image_path, text_prompt, output_dir))
        return make_segmentation_result(self.image_path, self.output_dir)


class FakeRequest:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, messages, **options):
        self.calls.append((messages, options))
        return next(self.responses)


def run_with_responses(tmp_path, responses, prompt="the object", **kwargs):
    image_path = tmp_path / "input.png"
    make_fixture_image(image_path)
    tool = FakeSegmentationTool(image_path, tmp_path)
    request = FakeRequest(responses)
    result = run_single_image_inference(
        image_path=str(image_path),
        text_prompt=prompt,
        llm_config={"name": "fake"},
        send_generate_request=request,
        segmentation_tool=tool,
        output_dir=str(tmp_path / "outputs"),
        **kwargs,
    )
    return result, tool, request


def test_segment_and_select_flow(tmp_path):
    result, tool, request = run_with_responses(
        tmp_path,
        [
            native_tool_response(
                "segment_phrase", {"text_prompt": "object"}, "call_segment"
            ),
            native_tool_response(
                "select_masks_and_return",
                {"final_answer_masks": [1]},
                "call_select",
            ),
        ],
        final_mask_output_dir=str(tmp_path / "final_masks"),
        verbose=False,
    )

    assert result["status"] == "success"
    assert len(tool.calls) == 1
    assert Path(result["output_json_path"]).exists()
    assert Path(result["output_image_path"]).exists()
    assert Path(result["final_mask_path"]).exists()

    first_tool_names = {
        item["function"]["name"] for item in request.calls[0][1]["tools"]
    }
    second_tool_names = {
        item["function"]["name"] for item in request.calls[1][1]["tools"]
    }
    assert first_tool_names == {"segment_phrase", "report_no_mask"}
    assert second_tool_names == {
        "segment_phrase",
        "examine_each_mask",
        "select_masks_and_return",
        "report_no_mask",
    }
    assert request.calls[0][1]["tool_choice"] == "required"
    assert request.calls[0][1]["parallel_tool_calls"] is False

    history = json.loads(Path(result["agent_history_path"]).read_text())
    assistant = next(item for item in history if item.get("tool_calls"))
    tool_message = next(item for item in history if item.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"] == tool_message["tool_call_id"]


def test_report_no_mask_does_not_call_segmentation_tool(tmp_path):
    result, tool, _ = run_with_responses(
        tmp_path,
        [native_tool_response("report_no_mask", {}, "call_none")],
        prompt="a unicorn",
    )

    assert result["status"] == "success"
    assert tool.calls == []
    output = json.loads(Path(result["output_json_path"]).read_text())
    assert output["pred_masks"] == []


def test_duplicate_prompt_is_not_sent_to_segmentation_tool(tmp_path):
    _, tool, _ = run_with_responses(
        tmp_path,
        [
            native_tool_response(
                "segment_phrase", {"text_prompt": "object"}, "call_segment"
            ),
            native_tool_response(
                "segment_phrase", {"text_prompt": "object"}, "call_duplicate"
            ),
            native_tool_response(
                "select_masks_and_return",
                {"final_answer_masks": [1]},
                "call_select",
            ),
        ],
    )

    assert [call[1] for call in tool.calls] == ["object"]


def test_examine_each_mask_accepts_and_returns_selected_mask(tmp_path):
    result, _, _ = run_with_responses(
        tmp_path,
        [
            native_tool_response(
                "segment_phrase", {"text_prompt": "object"}, "call_segment"
            ),
            native_tool_response("examine_each_mask", {}, "call_examine"),
            LLMResponse(
                content="<think>the mask matches</think><verdict>Accept</verdict>"
            ),
            native_tool_response(
                "select_masks_and_return",
                {"final_answer_masks": [1]},
                "call_select",
            ),
        ],
    )

    assert result["status"] == "success"
    history = json.loads(Path(result["agent_history_path"]).read_text())
    assert count_images(history) <= 2


def test_generation_limit_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="maximum number"):
        run_with_responses(
            tmp_path,
            [
                native_tool_response(
                    "segment_phrase", {"text_prompt": "object"}, "call_segment"
                )
            ],
            max_generations=0,
        )


@pytest.mark.parametrize(
    ("responses", "error"),
    [
        ([LLMResponse(content="no call")], "exactly one native tool call"),
        (
            [
                LLMResponse(
                    content=None,
                    tool_calls=(
                        FunctionToolCall("one", "segment_phrase", "{}"),
                        FunctionToolCall("two", "report_no_mask", "{}"),
                    ),
                )
            ],
            "exactly one native tool call",
        ),
        (
            [native_tool_response("segment_phrase", "{not-json")],
            "Invalid JSON arguments",
        ),
        ([native_tool_response("unknown", {})], "Unknown tool call"),
        (
            [native_tool_response("segment_phrase", {"text_prompt": "x", "extra": 1})],
            "must contain exactly",
        ),
    ],
)
def test_invalid_native_tool_calls_raise_clear_errors(tmp_path, responses, error):
    with pytest.raises((TypeError, ValueError), match=error):
        run_with_responses(tmp_path, responses)


@pytest.mark.parametrize("selected", [[1, 1], [2], [], [True]])
def test_invalid_mask_selection_is_rejected(tmp_path, selected):
    responses = [
        native_tool_response(
            "segment_phrase", {"text_prompt": "object"}, "call_segment"
        ),
        native_tool_response(
            "select_masks_and_return",
            {"final_answer_masks": selected},
            "call_select",
        ),
    ]
    with pytest.raises(ValueError):
        run_with_responses(tmp_path, responses)

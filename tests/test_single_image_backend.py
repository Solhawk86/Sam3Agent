import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam3_agent.tools import Sam3Tool


class FakeInteractiveModel:
    '''模拟 SAM3 交互预测并记录原生参数。'''

    def __init__(self, height: int, width: int):
        '''保存图片尺寸并启用假的交互头。'''

        self.height = height
        self.width = width
        self.inst_interactive_predictor = object()
        self.calls = []

    def predict_inst(self, state, **kwargs):
        '''按 multimask 参数返回一个或三个候选。'''

        self.calls.append(kwargs)
        count = 3 if kwargs["multimask_output"] else 1
        masks = np.zeros((count, self.height, self.width), dtype=np.uint8)
        scores = np.asarray([0.2, 0.9, 0.5][:count], dtype=np.float32)
        logits = np.zeros((count, 4, 4), dtype=np.float32)
        for index in range(count):
            masks[index, 1 : 4 + index, 2 : 6 + index] = 1
            logits[index].fill(index + 1)
        return masks, scores, logits


class FakeProcessor:
    '''模拟概念和交互式单图 processor。'''

    def __init__(self, height: int, width: int):
        '''创建固定预测结果和调用记录。'''

        self.height = height
        self.width = width
        self.model = FakeInteractiveModel(height, width)
        self.calls = []

    def set_image(self, image):
        '''记录图片设置并返回固定概念预测状态。'''

        self.calls.append(("image", image.size))
        mask = torch.zeros((1, 1, self.height, self.width), dtype=torch.bool)
        mask[:, :, 5:25, 10:30] = True
        return {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "masks": mask,
            "scores": torch.tensor([0.8]),
        }

    def set_text_prompt(self, state, prompt):
        '''记录文本提示并保留固定预测状态。'''

        self.calls.append(("text", prompt))
        return state

    def add_geometric_prompt(self, box, label, state):
        '''记录已转换的几何提示和正负标签。'''

        self.calls.append(("box", box, label))
        return state


def make_backend(tmp_path: Path, width: int = 100, height: int = 50):
    '''创建带假 processor 的后端和测试图片。'''

    image_path = tmp_path / "input.png"
    Image.new("RGB", (width, height), (240, 240, 240)).save(image_path)
    backend = Sam3Tool(device="cpu", enable_inst_interactivity=True)
    backend._processor = FakeProcessor(height, width)
    return backend, image_path


def test_visual_examples_convert_xyxy_and_preserve_prompt_order(tmp_path):
    '''验证视觉框转换为归一化 CXCYWH 并按输入顺序提交。'''

    backend, image_path = make_backend(tmp_path)
    result = backend.segment_visual_examples(
        str(image_path),
        [
            {"box": [10, 5, 30, 25], "label": "positive"},
            {"box": [50, 10, 70, 20], "label": "negative"},
        ],
        str(tmp_path / "output"),
    )

    assert backend.processor.calls == [
        ("image", (100, 50)),
        ("box", [0.2, 0.3, 0.2, 0.4], True),
        ("box", [0.6, 0.3, 0.2, 0.2], False),
    ]
    assert np.allclose(result.data["pred_boxes"], [[0.1, 0.1, 0.2, 0.4]])
    assert Path(result.json_path).is_file()
    assert Path(result.image_path).is_file()


def test_text_is_applied_before_visual_examples(tmp_path):
    '''验证联合概念分割先设置文本再追加正负样例。'''

    backend, image_path = make_backend(tmp_path)
    backend.segment_phrase_with_visual_examples(
        str(image_path),
        "crab",
        [{"box": [10, 5, 30, 25], "label": "negative"}],
        str(tmp_path / "output"),
    )

    assert backend.processor.calls[:3] == [
        ("image", (100, 50)),
        ("text", "crab"),
        ("box", [0.2, 0.3, 0.2, 0.4], False),
    ]


def test_single_foreground_point_keeps_all_sorted_candidates(tmp_path):
    '''验证单点使用多候选并保留重叠备选结果。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    result = backend.segment_instance_with_foreground_points(
        str(image_path),
        [[2, 3]],
        str(tmp_path / "output"),
    )

    call = backend.processor.model.calls[0]
    assert call["multimask_output"] is True
    assert call["point_coords"].tolist() == [[2.0, 3.0]]
    assert call["point_labels"].tolist() == [1]
    assert result.data["pred_scores"] == pytest.approx([0.9, 0.5, 0.2])
    assert len(result.data["pred_masks"]) == 3
    assert "kept_indices" not in result.data
    handles = result.data["refinement_handles"]
    assert [item["mask_number"] for item in handles] == [1, 2, 3]
    with np.load(handles[0]["refinement_handle"], allow_pickle=False) as archive:
        assert archive["mask_logits"].shape == (1, 4, 4)
        assert np.all(archive["mask_logits"] == 2)


def test_background_refinement_loads_logits_and_accumulates_points(tmp_path):
    '''验证背景点调用加载所选候选 logits 并累计正负点。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    first = backend.segment_instance_with_foreground_points(
        str(image_path),
        [[2, 3]],
        str(tmp_path / "output"),
    )
    handle = first.data["refinement_handles"][0]["refinement_handle"]
    refined = backend.refine_instance_with_background_points(
        str(image_path),
        [[8, 7]],
        handle,
        str(tmp_path / "output"),
    )

    call = backend.processor.model.calls[1]
    assert call["multimask_output"] is False
    assert call["point_coords"].tolist() == [[2.0, 3.0], [8.0, 7.0]]
    assert call["point_labels"].tolist() == [1, 0]
    assert call["mask_input"].shape == (1, 4, 4)
    assert np.all(call["mask_input"] == 2)
    assert refined.data["prompt_data"]["foreground_points"] == [[2.0, 3.0]]
    assert refined.data["prompt_data"]["background_points"] == [[8, 7]]
    assert len(refined.data["refinement_handles"]) == 1


def test_foreground_refinement_loads_logits_and_accumulates_points(tmp_path):
    '''验证前景点工具可用候选句柄追加新的正点。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    first = backend.segment_instance_with_foreground_points(
        str(image_path),
        [[2, 3]],
        str(tmp_path / "output"),
    )
    handle = first.data["refinement_handles"][1]["refinement_handle"]
    refined = backend.segment_instance_with_foreground_points(
        str(image_path),
        [[5, 6]],
        str(tmp_path / "output"),
        refinement_handle=handle,
    )

    call = backend.processor.model.calls[1]
    assert call["multimask_output"] is False
    assert call["point_coords"].tolist() == [[2.0, 3.0], [5.0, 6.0]]
    assert call["point_labels"].tolist() == [1, 1]
    assert np.all(call["mask_input"] == 3)
    assert refined.data["prompt_data"]["foreground_points"] == [
        [2.0, 3.0],
        [5, 6],
    ]


def test_box_and_labeled_points_map_to_native_predict_arguments(tmp_path):
    '''验证框提示与带标签点正确映射到 predict_inst。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    backend.segment_instance_with_box(
        str(image_path),
        [1, 2, 10, 9],
        str(tmp_path / "output"),
    )
    backend.segment_instance_with_points_and_box(
        str(image_path),
        [1, 2, 10, 9],
        [
            {"point": [3, 4], "label": "foreground"},
            {"point": [8, 7], "label": "background"},
        ],
        str(tmp_path / "output"),
    )

    box_call, combined_call = backend.processor.model.calls
    assert box_call["box"].tolist() == [1.0, 2.0, 10.0, 9.0]
    assert box_call["point_coords"] is None
    assert combined_call["point_coords"].tolist() == [[3.0, 4.0], [8.0, 7.0]]
    assert combined_call["point_labels"].tolist() == [1, 0]
    assert combined_call["box"].tolist() == [1.0, 2.0, 10.0, 9.0]


def test_repeated_interactive_arguments_do_not_overwrite_outputs(tmp_path):
    '''验证相同参数的重复调用使用递增后缀保留旧产物。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    first = backend.segment_instance_with_box(
        str(image_path),
        [1, 2, 10, 9],
        str(tmp_path / "output"),
    )
    second = backend.segment_instance_with_box(
        str(image_path),
        [1, 2, 10, 9],
        str(tmp_path / "output"),
    )

    assert first.json_path != second.json_path
    assert Path(first.json_path).is_file()
    assert Path(second.json_path).is_file()


@pytest.mark.parametrize(
    ("method", "arguments", "error"),
    [
        (
            "segment_visual_examples",
            [[{"box": [1, 1, 5, 5], "label": "negative"}]],
            "positive example",
        ),
        (
            "segment_instance_with_foreground_points",
            [[[12, 3]]],
            "inside image bounds",
        ),
        (
            "segment_instance_with_box",
            [[5, 1, 2, 8]],
            "must satisfy",
        ),
    ],
)
def test_backend_rejects_invalid_semantic_coordinates(
    tmp_path,
    method,
    arguments,
    error,
):
    '''验证后端拒绝缺少正样例、越界点和反向框。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    with pytest.raises(ValueError, match=error):
        getattr(backend, method)(
            str(image_path),
            *arguments,
            str(tmp_path / "output"),
        )


def test_refinement_handle_rejects_paths_outside_current_image(tmp_path):
    '''验证句柄不能读取当前图片输出目录之外的文件。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    outside = tmp_path / "outside.npz"
    np.savez_compressed(outside, mask_logits=np.zeros((1, 4, 4)), metadata="{}")

    with pytest.raises(ValueError, match="inside the current image output"):
        backend.refine_instance_with_background_points(
            str(image_path),
            [[1, 1]],
            str(outside),
            str(tmp_path / "output"),
        )


def test_refinement_handle_rejects_corrupted_archive(tmp_path):
    '''验证当前输出目录内损坏的句柄也会被拒绝。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    handle_dir = tmp_path / "output" / image_path.stem / "refinement"
    handle_dir.mkdir(parents=True)
    corrupted = handle_dir / "corrupted.npz"
    corrupted.write_bytes(b"not a numpy archive")

    with pytest.raises(ValueError):
        backend.refine_instance_with_background_points(
            str(image_path),
            [[1, 1]],
            str(corrupted),
            str(tmp_path / "output"),
        )


def test_refinement_handle_rejects_different_image_metadata(tmp_path):
    '''验证即使句柄路径合法也不能跨图片复用元数据。'''

    backend, image_path = make_backend(tmp_path, width=12, height=10)
    first = backend.segment_instance_with_box(
        str(image_path),
        [1, 1, 8, 8],
        str(tmp_path / "output"),
    )
    handle = Path(first.data["refinement_handles"][0]["refinement_handle"])
    with np.load(handle, allow_pickle=False) as archive:
        logits = archive["mask_logits"].copy()
        metadata = json.loads(str(archive["metadata"].item()))
    metadata["image_path"] = str(tmp_path / "another.png")
    np.savez_compressed(
        handle,
        mask_logits=logits,
        metadata=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="different image"):
        backend.refine_instance_with_background_points(
            str(image_path),
            [[1, 1]],
            str(handle),
            str(tmp_path / "output"),
        )

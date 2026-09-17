'''候选叠加与审核拼板的视觉结构测试。'''

import colorsys

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from sam3_agent.segmentation_memory import rendering
from sam3_agent.segmentation_memory.models import Candidate, SegmentationMemory


def make_memory(tmp_path, width=100, height=100):
    '''创建带纯黑原图的最小渲染状态。'''

    image_path = tmp_path / "image.png"
    Image.new("RGB", (width, height), "black").save(image_path)
    return SegmentationMemory(str(image_path), "object", width, height, "run")


def encode_mask(mask):
    '''将测试二值图编码为候选使用的 COCO RLE 字符串。'''

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return encoded["counts"].decode("ascii")


def test_render_candidates_uses_light_fill_and_strong_outline(tmp_path):
    '''主面板保留低透明填充，并用同色高对比轮廓强调边界。'''

    memory = make_memory(tmp_path)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    candidate = Candidate("m1", "t1", "box", encode_mask(mask), [], 0.9, 3600)

    rendered = np.asarray(rendering.render_candidates(memory, [candidate]))
    interior = rendered[70, 70]
    boundary = rendered[50, 20]
    outside = rendered[10, 10]
    expected_color = (
        np.asarray(colorsys.hsv_to_rgb((1 * 0.618) % 1, 0.8, 1)) * 255
    )

    assert np.array_equal(outside, [0, 0, 0])
    assert np.allclose(interior, expected_color * rendering.MASK_FILL_ALPHA, atol=1)
    assert boundary.max() > interior.max() * 3


def test_render_board_adds_gutter_and_status_counts(tmp_path, monkeypatch):
    '''双列拼板使用显式分隔带，并在状态标题中保留候选数量。'''

    memory = make_memory(tmp_path, width=80, height=40)
    memory.candidates["m1"] = Candidate(
        "m1", "t1", "box", "unused", [], 0.9, 0, "pending"
    )
    titles = []

    def fake_render_candidates(_memory, _candidates):
        '''返回固定尺寸面板，避免测试依赖候选 RLE。'''

        return Image.new("RGB", (80, 40), "white")

    def fake_caption(image, title):
        '''记录标题并保持面板尺寸不变。'''

        titles.append(title)
        return image

    monkeypatch.setattr(rendering, "render_candidates", fake_render_candidates)
    monkeypatch.setattr(rendering, "_caption", fake_caption)
    board = rendering.render_board(memory, [])
    gutter = max(16, round(memory.width * 0.025))
    pixels = np.asarray(board)

    assert titles == ["ACCEPTED (0)", "PENDING (1)"]
    assert board.size == (memory.width * 2 + gutter, memory.height)
    assert np.all(pixels[:, memory.width : memory.width + gutter] == (72, 72, 72))

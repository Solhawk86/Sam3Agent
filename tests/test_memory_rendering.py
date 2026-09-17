'''候选叠加与审核拼板的视觉结构测试。'''

import colorsys

import numpy as np
import pycocotools.mask as mask_utils
import pytest
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


@pytest.mark.parametrize(
    "size,expected",
    [
        ((200, 400), (256, 512)), ((400, 200), (512, 256)),
        ((256, 512), (256, 512)), ((512, 256), (512, 256)),
        ((512, 512), (512, 512)), ((300, 900), (300, 900)),
        ((900, 300), (900, 300)), ((600, 800), (600, 800)),
        ((1, 400), (1, 512)), ((333, 400), (426, 512)),
    ],
)
def test_closeup_resize_preserves_large_images(size, expected):
    '''小图按最长边放大，边界和大图逐像素保留。'''

    pixels = np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    resized = rendering._resize_closeup(image)
    assert resized.size == expected
    if max(size) >= 512:
        np.testing.assert_array_equal(np.asarray(resized), pixels)
    else:
        np.testing.assert_array_equal(
            np.asarray(resized), np.asarray(image.resize(expected, Image.Resampling.LANCZOS))
        )


@pytest.mark.parametrize("pending_count", [0, 1, 4, 5, 13])
def test_pages_include_pending_and_prioritize_inspections(tmp_path, monkeypatch, pending_count):
    '''显式检查先于数字排序的待审核项，分页和可见状态不会累积。'''

    memory = make_memory(tmp_path)
    # 逆序插入以验证排序不依赖字典或字符串顺序。
    for index in range(pending_count, 0, -1):
        memory.candidates[f"m{index}"] = Candidate(
            f"m{index}", "t1", "text", "unused", [], 0.9, 1
        )
    memory.candidates["m99"] = Candidate("m99", "t1", "box", "unused", [], 0.8, 1, "rejected")
    memory.candidates["m100"] = Candidate("m100", "t1", "box", "unused", [], 0.7, 1, "accepted")
    rendered_ids, titles = [], []
    original_caption = rendering._caption

    def fake_zoom(object_data, original, **kwargs):
        '''记录局部图的真实渲染顺序。'''

        rendered_ids.append(object_data["labels"][0]["noun_phrase"])
        return Image.new("RGB", (100, 100), "red"), "#ff0000"

    def capture_caption(image, title):
        '''记录页面和候选标题，同时使用真实标题排版。'''

        titles.append(title)
        return original_caption(image, title)

    monkeypatch.setattr(rendering, "render_zoom_in", fake_zoom)
    monkeypatch.setattr(rendering, "_caption", capture_caption)
    pending_ids = [f"m{index}" for index in range(1, pending_count + 1)]
    requested = ["m99"] + pending_ids[-1:]
    expected = list(dict.fromkeys(requested + pending_ids))
    pages = rendering.render_closeup_pages(memory, requested + requested)
    assert rendered_ids == expected == memory.inspection_ids
    assert set(memory.visible_ids) == set(expected) | {"m100"}
    assert len(pages) == (len(expected) + 3) // 4
    for index in range(len(pages)):
        assert f"Page {index + 1}/{len(pages)}: {', '.join(expected[index * 4:index * 4 + 4])}" in titles
    assert "m99 / box / rejected / score=0.800" in titles

    pages = rendering.render_closeup_pages(memory, [])
    assert memory.inspection_ids == pending_ids
    assert set(memory.visible_ids) == set(pending_ids) | {"m100"}
    assert len(pages) == (pending_count + 3) // 4
    if pending_count % 4:
        assert pages[-1].getpixel((pages[-1].width - 1, pages[-1].height - 1)) == (255, 255, 255)


def test_page_preserves_mixed_size_pixels_and_empty_slot():
    '''大图在四宫格内不裁切不缩放，狭长图居中且末格留白。'''

    sizes = [(600, 800), (300, 900), (900, 300)]
    colors = [(210, 10, 20), (10, 210, 20), (10, 20, 210)]
    crops = [Image.new("RGB", size, color) for size, color in zip(sizes, colors)]
    page = rendering._closeup_page(crops, ["m1", "m2", "m3"], "Page 1/1: m1, m2, m3")
    pixels = np.asarray(page)
    # 每张图的纯色像素包围盒等于其原始尺寸，不会受标题和留白影响。
    positions = []
    for (width, height), color in zip(sizes, colors):
        rows, columns = np.nonzero(np.all(pixels == color, axis=2))
        assert len(rows) == width * height
        assert columns.max() - columns.min() + 1 == width
        assert rows.max() - rows.min() + 1 == height
        positions.append((columns.min(), rows.min()))
    assert positions[0][0] < positions[1][0]
    assert positions[0][1] < positions[2][1]
    assert page.getpixel((page.width - 100, page.height - 100)) == (255, 255, 255)
    assert page.width == 900 + 16 + 512


def test_caption_wraps_without_touching_image():
    '''长标题换行后仍完整保留底部图像。'''

    image = Image.new("RGB", (512, 80), "red")
    short = rendering._caption(image, "short")
    long = rendering._caption(image, "mask_id / branch / status / score " * 12)
    assert long.height > short.height
    np.testing.assert_array_equal(np.asarray(long)[-80:], np.asarray(image))
    assert long.width == image.width


@pytest.mark.parametrize("size", [(800, 600), (1600, 900), (900, 1600)])
def test_overview_counts_and_size_limit(tmp_path, monkeypatch, size):
    '''概览包含状态计数且完整画布长边不超过上限。'''

    memory = make_memory(tmp_path, *size)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    for index, status in enumerate(["accepted", "pending", "rejected"], 1):
        memory.candidates[f"m{index}"] = Candidate(
            f"m{index}", "t1", "text", encode_mask(mask), [], 0.9, 10000, status
        )
    titles = []
    original_caption = rendering._caption

    def capture_caption(image, title):
        '''记录概览计数并保留实际排版。'''

        titles.append(title)
        return original_caption(image, title)

    monkeypatch.setattr(rendering, "_caption", capture_caption)
    overview = rendering.render_overview(memory)
    assert max(overview.size) <= 1280
    assert titles == ["ACCEPTED (1) / PENDING (1)"]
    if size == (800, 600):
        assert overview.width == 800
        assert overview.getpixel((125, overview.height - 600 + 125)) == (0, 0, 0)


def test_zero_area_closeup_uses_original(tmp_path):
    '''零面积候选仍可使用原图生成审核分页。'''

    memory = make_memory(tmp_path, 600, 800)
    memory.candidates["m1"] = Candidate("m1", "t1", "box", "unused", [], 0.0, 0)
    pages = rendering.render_closeup_pages(memory, [])
    pixels = np.asarray(pages[0])
    # 标题也有黑色像素，完整原图区域应至少保留原像素数量。
    assert np.count_nonzero(np.all(pixels == (0, 0, 0), axis=2)) >= 600 * 800
    assert memory.inspection_ids == ["m1"]

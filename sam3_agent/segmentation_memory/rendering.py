'''使用稳定候选编号绘制当前结果和局部审核视图。'''

import colorsys

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont

from ..helpers.zoom_in import render_zoom_in
from .models import Candidate, SegmentationMemory


MASK_FILL_ALPHA = 0.15
PANEL_GUTTER_COLOR = (72, 72, 72)


def _font(size: int) -> ImageFont.ImageFont:
    '''优先使用可缩放字体，缺失时退回 Pillow 自带字体。'''

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _caption(image: Image.Image, title: str) -> Image.Image:
    '''添加英文状态和稳定 ID 标题，避免依赖系统中文字体。'''

    height = max(28, image.width // 32)
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text(
        (6, 4), title, fill="black", font=_font(max(12, height - 10))
    )
    return canvas


def render_candidates(
    memory: SegmentationMemory, candidates: list[Candidate]
) -> Image.Image:
    '''叠加候选及同色稳定标签，不使用候选列表位置编号。'''

    with Image.open(memory.image_path) as original:
        pixels = np.asarray(original.convert("RGB")).copy()
    labels = []
    for candidate in candidates:
        binary = mask_utils.decode(
            {
                "size": [memory.height, memory.width],
                "counts": candidate.rle,
            }
        ).astype(bool)
        color = (
            np.asarray(
                colorsys.hsv_to_rgb((int(candidate.mask_id[1:]) * 0.618) % 1, 0.8, 1)
            )
            * 255
        )
        color_uint8 = tuple(map(int, color))
        pixels[binary] = (
            pixels[binary] * (1 - MASK_FILL_ALPHA) + color * MASK_FILL_ALPHA
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            pixels,
            contours,
            -1,
            color_uint8,
            max(2, round(min(memory.width, memory.height) / 300)),
            lineType=cv2.LINE_AA,
        )
        rows, columns = np.nonzero(binary)
        if len(rows):
            middle = len(rows) // 2
            position = (int(columns[middle]), int(rows[middle]))
        else:
            position = (4, 4 + len(labels) * 20)
        labels.append(
            (
                position,
                f"{candidate.mask_id}/{candidate.branch}",
                color_uint8,
            )
        )
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = _font(max(12, min(memory.width, memory.height) // 40))
    for position, text, color in labels:
        bounds = draw.textbbox(position, text, font=font)
        draw.rectangle(bounds, fill=color, outline="white")
        draw.text(
            position, text, font=font, fill="white", stroke_width=1, stroke_fill="black"
        )
    return image


def render_board(memory: SegmentationMemory, inspect_ids: list[str]) -> Image.Image:
    '''合成全部有效和待审核候选，以及最多四个局部检查面板。'''

    panels = []
    visible_ids = []
    for status in ("accepted", "pending"):
        candidates = [
            item for item in memory.candidates.values() if item.status == status
        ]
        visible_ids.extend(item.mask_id for item in candidates)
        panels.append(
            _caption(
                render_candidates(memory, candidates),
                f"{status.upper()} ({len(candidates)})",
            )
        )
    with Image.open(memory.image_path) as original:
        for mask_id in inspect_ids:
            candidate = memory.candidates[mask_id]
            if candidate.area:
                crop, _ = render_zoom_in(
                    {
                        "labels": [{"noun_phrase": mask_id}],
                        "segmentation": {
                            "size": [memory.height, memory.width],
                            "counts": candidate.rle,
                        },
                    },
                    original,
                    show_text=False,
                )
            else:
                crop = original.convert("RGB")
            crop = crop.convert("RGB")
            target_height = max(1, round(crop.height * memory.width / crop.width))
            crop = crop.resize((memory.width, target_height))
            panels.append(
                _caption(crop, f"{mask_id} / {candidate.branch} / {candidate.status}")
            )
            if mask_id not in visible_ids:
                visible_ids.append(mask_id)
    rows = [panels[index : index + 2] for index in range(0, len(panels), 2)]
    heights = [max(panel.height for panel in row) for row in rows]
    gutter = max(16, round(memory.width * 0.025))
    board = Image.new(
        "RGB",
        (memory.width * 2 + gutter, sum(heights)),
        "white",
    )
    ImageDraw.Draw(board).rectangle(
        (memory.width, 0, memory.width + gutter - 1, board.height - 1),
        fill=PANEL_GUTTER_COLOR,
    )
    y = 0
    for row, height in zip(rows, heights):
        for column, panel in enumerate(row):
            x = 0 if column == 0 else memory.width + gutter
            board.paste(panel, (x, y))
        y += height
    memory.visible_ids = visible_ids
    return board

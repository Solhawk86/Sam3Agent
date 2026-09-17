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
CLOSEUP_MIN_SIZE = 512
CLOSEUPS_PER_PAGE = 4
PAGE_GUTTER = 16


def _font(size: int) -> ImageFont.ImageFont:
    '''优先使用可缩放字体，缺失时退回 Pillow 自带字体。'''

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _caption(image: Image.Image, title: str) -> Image.Image:
    '''在图像外添加可换行标题，保留图像内容和尺寸。'''

    font = _font(18)
    padding = min(6, max(0, (image.width - 1) // 2))
    available = max(1, image.width - padding * 2)
    lines = []
    line = ""
    for character in title:
        if line and font.getlength(line + character) > available:
            lines.append(line)
            line = ""
        line += character
    lines.append(line)
    line_height = 24
    height = len(lines) * line_height + 12
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((padding, 4 + index * line_height), line, fill="black", font=font)
    return canvas


def render_candidates(
    memory: SegmentationMemory,
    candidates: list[Candidate],
    *,
    fill_alpha: float = MASK_FILL_ALPHA,
    show_status: bool = False,
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
            pixels[binary] * (1 - fill_alpha) + color * fill_alpha
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
                f"{candidate.mask_id}/{candidate.branch}"
                + (f"/{candidate.status}" if show_status else ""),
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


def render_overview(memory: SegmentationMemory) -> Image.Image:
    '''在单张原图上绘制当前候选轮廓和状态，并限制概览长边。'''

    candidates = [
        item for item in memory.candidates.values()
        if item.status in {"accepted", "pending"}
    ]
    image = render_candidates(memory, candidates, fill_alpha=0, show_status=True)
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    # 小原图只补充标题所需留白，不放大图像像素。
    if image.width < 256:
        canvas = Image.new("RGB", (256, image.height), "white")
        canvas.paste(image, ((256 - image.width) // 2, 0))
        image = canvas
    accepted = sum(item.status == "accepted" for item in candidates)
    pending = len(candidates) - accepted
    image = _caption(image, f"ACCEPTED ({accepted}) / PENDING ({pending})")
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    return image


def _resize_closeup(crop: Image.Image) -> Image.Image:
    '''只将最长边不足 512 的局部合成图等比放大。'''

    crop = crop.convert("RGB")
    longest = max(crop.size)
    if longest < CLOSEUP_MIN_SIZE:
        scale = CLOSEUP_MIN_SIZE / longest
        crop = crop.resize(
            tuple(max(1, round(side * scale)) for side in crop.size),
            Image.Resampling.LANCZOS,
        )
    return crop


def _closeup_page(
    crops: list[Image.Image], titles: list[str], page_title: str
) -> Image.Image:
    '''将至多四张局部图原尺寸居中放入自适应四宫格。'''

    widths = [
        max([CLOSEUP_MIN_SIZE] + [crop.width for crop in crops[column::2]])
        for column in range(2)
    ]
    heights = [
        max([CLOSEUP_MIN_SIZE] + [crop.height for crop in crops[row * 2:row * 2 + 2]])
        for row in range(2)
    ]
    panels = []
    for index, (crop, title) in enumerate(zip(crops, titles)):
        width, height = widths[index % 2], heights[index // 2]
        cell = Image.new("RGB", (width, height), "white")
        cell.paste(crop, ((width - crop.width) // 2, (height - crop.height) // 2))
        panels.append(_caption(cell, title))
    row_heights = [
        max([heights[row]] + [panel.height for panel in panels[row * 2:row * 2 + 2]])
        for row in range(2)
    ]
    page = Image.new(
        "RGB", (sum(widths) + PAGE_GUTTER, sum(row_heights) + PAGE_GUTTER), "white"
    )
    draw = ImageDraw.Draw(page)
    draw.rectangle(
        (widths[0], 0, widths[0] + PAGE_GUTTER - 1, page.height - 1),
        fill=PANEL_GUTTER_COLOR,
    )
    draw.rectangle(
        (0, row_heights[0], page.width - 1, row_heights[0] + PAGE_GUTTER - 1),
        fill=PANEL_GUTTER_COLOR,
    )
    for index, panel in enumerate(panels):
        x = 0 if index % 2 == 0 else widths[0] + PAGE_GUTTER
        y = 0 if index < 2 else row_heights[0] + PAGE_GUTTER
        page.paste(panel, (x, y))
    return _caption(page, page_title)


def render_closeup_pages(
    memory: SegmentationMemory, inspect_ids: list[str]
) -> list[Image.Image]:
    '''显式检查优先，每四个候选生成一页，并刷新本轮可见编号。'''

    pending_ids = sorted(
        (item.mask_id for item in memory.candidates.values() if item.status == "pending"),
        key=lambda mask_id: int(mask_id[1:]),
    )
    inspection_ids = list(dict.fromkeys(inspect_ids + pending_ids))
    pages = []
    page_count = (len(inspection_ids) + CLOSEUPS_PER_PAGE - 1) // CLOSEUPS_PER_PAGE
    with Image.open(memory.image_path) as original:
        for start in range(0, len(inspection_ids), CLOSEUPS_PER_PAGE):
            page_ids = inspection_ids[start:start + CLOSEUPS_PER_PAGE]
            crops, titles = [], []
            for mask_id in page_ids:
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
                crops.append(_resize_closeup(crop))
                titles.append(
                    f"{mask_id} / {candidate.branch} / {candidate.status} / "
                    f"score={candidate.score:.3f}"
                )
            pages.append(
                _closeup_page(
                    crops, titles,
                    f"Page {len(pages) + 1}/{page_count}: {', '.join(page_ids)}",
                )
            )
    overview_ids = [
        item.mask_id for item in memory.candidates.values()
        if item.status in {"accepted", "pending"}
    ]
    memory.visible_ids = list(dict.fromkeys(overview_ids + inspection_ids))
    memory.inspection_ids = inspection_ids
    return pages

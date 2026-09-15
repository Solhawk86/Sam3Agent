from pathlib import Path
from typing import Optional


def apply_slice(
    image_paths: list[Path],
    start_index: int,
    limit: Optional[int],
) -> list[Path]:
    """对图片列表应用起始下标和数量限制。"""
    image_paths = image_paths[start_index:]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def iter_images(
    image_dir: Path,
    pattern: str,
    start_index: int,
    limit: Optional[int],
) -> list[Path]:
    """按 glob 规则枚举图片，并应用起始下标和数量限制。"""
    return apply_slice(sorted(Path(image_dir).glob(pattern)), start_index, limit)


def load_image_list(
    image_list: Path,
    workspace_root: Path,
    start_index: int,
    limit: Optional[int],
) -> list[Path]:
    """从文本文件读取图片路径列表，并应用起始下标和数量限制。"""
    image_paths = []
    with Path(image_list).open("r") as handle:
        for line in handle:
            path_text = line.strip()
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            if not path.is_absolute():
                path = workspace_root / path
            image_paths.append(path)
    return apply_slice(image_paths, start_index, limit)

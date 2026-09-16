'''单图工具与批量调度共用的坐标校验。'''

import math
from typing import Any


def validate_number(value: Any, field_name: str) -> float:
    '''校验有限实数，避免布尔值和非有限坐标进入后端。'''

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must contain only numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return number


def validate_box(
    box: Any, width: int, height: int, field_name: str = "box"
) -> list[float]:
    '''校验原图像素 XYXY 框，返回统一浮点坐标。'''

    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{field_name} must be a four-number [x1, y1, x2, y2] array")
    values = [validate_number(value, field_name) for value in box]
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"{field_name} must satisfy 0 <= x1 < x2 <= {width} and "
            f"0 <= y1 < y2 <= {height}"
        )
    return values

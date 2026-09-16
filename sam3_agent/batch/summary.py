import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 友好的 JSON 格式写出结构化结果。"""
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_summary_path(output_dir: Path) -> Path:
    """返回本次批量运行摘要文件的固定路径。"""
    return Path(output_dir) / "run_summary.json"


def load_summary(summary_path: Path) -> dict[str, Any]:
    """读取已有运行摘要；不存在或格式异常时返回空字典。"""
    if not Path(summary_path).exists():
        return {}
    try:
        payload = json.load(Path(summary_path).open("r"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def update_summary(
    summary_path: Path,
    image_name: str,
    success: bool,
    llm_requests: int,
    elapsed_sec: float,
    status: str | None = None,
    statistics: dict[str, Any] | None = None,
) -> None:
    """更新单张图片的运行状态、LLM 请求次数和耗时。"""
    summary = load_summary(summary_path)
    summary[image_name] = {
        "success": success,
        "status": status or ("success" if success else "error"),
        "statistics": statistics or {},
        "llm_requests": llm_requests,
        "elapsed_sec": round(elapsed_sec, 3),
    }
    write_json(summary_path, summary)


def write_error_file(result_dir: Path, error_text: str) -> None:
    """把单张图片处理失败的异常堆栈写入结果目录。"""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "error.txt").write_text(error_text)

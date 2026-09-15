import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    """递归展开配置中的 ${ENV_NAME} 环境变量占位符。"""
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_config(config_path: str | Path, config_dir: Path) -> tuple[dict[str, Any], Path]:
    """读取 YAML 配置文件，并完成环境变量占位符替换。"""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    with path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    return expand_env(config), path


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """安全读取配置分组；分组不存在或不是字典时返回空字典。"""
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_path(
    value: Optional[str | Path],
    base_dir: Path,
) -> Optional[Path]:
    """把配置或命令行中的路径转换为基于工作区的 Path。"""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path

import os
import sys

from .schema import RuntimeConfig


def apply_environment(environment: dict) -> None:
    """把 YAML 中 environment 分组写入当前进程环境变量。"""
    for key, value in environment.items():
        if value is not None:
            os.environ[str(key)] = str(value)


def configure_sam3_import(runtime: RuntimeConfig) -> None:
    """配置 SAM3 源码路径，保证 sam3 包可被当前入口导入。"""
    sam3_root = runtime.sam3_root
    if sam3_root is not None and str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    if runtime.chdir_sam3_root and sam3_root is not None:
        os.chdir(sam3_root)

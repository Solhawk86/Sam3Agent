"""Independent SAM3 agent package."""

__all__ = ["agent_inference", "run_single_image_inference"]


def __getattr__(name):
    """按需导入较重的推理入口，避免普通 CLI 操作加载视觉依赖。"""
    if name == "agent_inference":
        from .agent_core import agent_inference

        return agent_inference
    if name == "run_single_image_inference":
        from .inference import run_single_image_inference

        return run_single_image_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

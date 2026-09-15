import re
from pathlib import Path

from sam3_agent.config.schema import PromptConfig


def extract_prompt(image_path: Path, config: PromptConfig) -> str:
    """按照 prompt 配置从图片文件名中提取文本提示词。"""
    match = re.match(config.regex, image_path.stem)
    if match is None and config.require_match:
        raise ValueError(f"Cannot extract prompt from filename: {image_path.name}")

    prompt = re.sub(config.regex, config.replacement, image_path.stem)
    prompt = prompt.replace("_", config.underscore_replacement)
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"Cannot extract prompt from filename: {image_path.name}")
    return prompt

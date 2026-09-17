import re
from pathlib import Path

from sam3_agent.config.schema import PromptConfig


def extract_prompt(image_path: Path, config: PromptConfig) -> str:
    '''从文件名提取提示词，并按配置将类别名规范化为可读词组。'''

    match = re.match(config.regex, image_path.stem)
    if match is None and config.require_match:
        raise ValueError(f"Cannot extract prompt from filename: {image_path.name}")

    prompt = re.sub(config.regex, config.replacement, image_path.stem)
    prompt = prompt.replace("_", config.underscore_replacement)
    if config.normalize_class_name:
        prompt = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", prompt)
        prompt = " ".join(prompt.lower().split())
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"Cannot extract prompt from filename: {image_path.name}")
    return prompt

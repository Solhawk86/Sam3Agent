from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class RuntimeConfig:
    """SAM3 模型与运行环境配置。"""

    sam3_root: Optional[Path]
    checkpoint_path: Optional[Path]
    device: Optional[str]
    confidence_threshold: float
    bpe_path: Optional[Path]
    load_from_hf: bool
    chdir_sam3_root: bool


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容 LLM API 配置。"""

    provider: str
    name: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    api_key_env: Optional[str]
    max_tokens: int
    extra_body: dict[str, Any]


@dataclass(frozen=True)
class InputConfig:
    """批量输入图片配置。"""

    image_dir: Optional[Path]
    image_list: Optional[Path]
    pattern: str
    start_index: int
    limit: Optional[int]


@dataclass(frozen=True)
class OutputConfig:
    """批量输出目录配置。"""

    output_dir: Path
    final_mask_dir: Optional[Path]


@dataclass(frozen=True)
class PromptConfig:
    """从文件名提取 prompt 的规则配置。"""

    regex: str
    replacement: str
    underscore_replacement: str
    require_match: bool


@dataclass(frozen=True)
class AgentConfig:
    """Agent 迭代与日志开关配置。"""

    max_generations: int
    debug: bool
    verbose: bool


@dataclass(frozen=True)
class RunConfig:
    """入口运行所需的完整配置快照。"""

    config_path: Path
    workspace_root: Path
    environment: dict[str, Any]
    runtime: RuntimeConfig
    llm: LLMConfig
    input: InputConfig
    output: OutputConfig
    prompt: PromptConfig
    agent: AgentConfig
    raw: dict[str, Any]

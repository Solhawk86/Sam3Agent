import argparse
from pathlib import Path

from sam3_agent.batch import run_batch
from sam3_agent.config import (
    apply_environment,
    build_run_config,
    configure_sam3_import,
    load_config,
)


ENTRY_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ENTRY_ROOT.parent
DEFAULT_CONFIG = ENTRY_ROOT / "agent_config.yaml"


def build_parser():
    """构建入口脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run SAM3 agent inference with YAML-injected API settings."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image-dir")
    parser.add_argument("--image-list")
    parser.add_argument("--output-dir")
    parser.add_argument("--final-mask-dir")
    parser.add_argument("--pattern")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-generations", type=int)
    parser.add_argument("--max-box-tasks-per-round", type=int)
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def main(argv=None):
    """读取配置并调用批量 runner。"""
    args = build_parser().parse_args(argv)
    raw_config, config_path = load_config(args.config, ENTRY_ROOT)
    apply_environment(raw_config.get("environment", {}))
    config = build_run_config(raw_config, config_path, WORKSPACE_ROOT, args=args)
    configure_sam3_import(config.runtime)
    run_batch(config)


if __name__ == "__main__":
    main()

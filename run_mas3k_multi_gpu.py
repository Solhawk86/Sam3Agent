#!/usr/bin/env python3
'''在多个 GPU 进程间拆分 MAS3K 图片并汇总输出。'''

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ENTRY_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ENTRY_ROOT.parent
DEFAULT_CONFIG = ENTRY_ROOT / "agent_config.yaml"
DEFAULT_IMAGE_DIR = WORKSPACE_ROOT / "MAS3K_test_small_100" / "Image"
DEFAULT_OUTPUT_DIR = ENTRY_ROOT / "result" / "agent_output_mas3k_test_small_100"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    '''构建多 GPU 批量启动参数。'''

    parser = argparse.ArgumentParser(
        description="Split MAS3K images across one SAM3 agent process per GPU."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-mask-dir", type=Path)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["2", "3"],
        help="GPU IDs, for example: --gpus 2 3 (comma-separated values also work).",
    )
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the first selected image on the first GPU.",
    )
    parser.add_argument("--max-generations", type=int)
    parser.add_argument("--max-box-tasks-per-round", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write shard lists and print commands without starting workers.",
    )
    return parser


def parse_gpu_ids(values: Sequence[str]) -> list[str]:
    '''解析空格或逗号分隔的 GPU 编号，并拒绝重复编号。'''

    gpu_ids = [item.strip() for value in values for item in value.split(",")]
    gpu_ids = [item for item in gpu_ids if item]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus must not contain duplicate GPU IDs")
    return gpu_ids


def select_images(
    image_dir: Path,
    pattern: str,
    start_index: int,
    limit: int | None,
) -> list[Path]:
    '''按稳定顺序枚举图片并应用全局切片。'''

    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path.resolve()
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    images = images[start_index:]
    if limit is not None:
        images = images[:limit]
    return images


def split_images(images: Sequence[Path], worker_count: int) -> list[list[Path]]:
    '''用轮询方式生成数量均衡且互不重叠的图片分片。'''

    return [list(images[index::worker_count]) for index in range(worker_count)]


def write_shard_list(path: Path, images: Sequence[Path]) -> None:
    '''写出供 run_agent.py 使用的绝对图片路径清单。'''

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{image}\n" for image in images)
    path.write_text(content, encoding="utf-8")


def build_worker_command(
    args: argparse.Namespace,
    gpu_id: str,
    shard_path: Path,
    summary_path: Path,
    final_mask_dir: Path,
) -> list[str]:
    '''构建单个 GPU worker 的 run_agent.py 命令。'''

    command = [
        sys.executable,
        str(ENTRY_ROOT / "run_agent.py"),
        "--config",
        str(args.config.resolve()),
        "--gpu",
        gpu_id,
        "--image-list",
        str(shard_path),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--final-mask-dir",
        str(final_mask_dir),
        "--summary-path",
        str(summary_path),
    ]
    if args.max_generations is not None:
        command.extend(["--max-generations", str(args.max_generations)])
    if args.max_box_tasks_per_round is not None:
        command.extend(
            ["--max-box-tasks-per-round", str(args.max_box_tasks_per_round)]
        )
    if args.debug:
        command.append("--debug")
    if args.verbose:
        command.append("--verbose")
    return command


def terminate_workers(workers: Sequence[subprocess.Popen]) -> None:
    '''终止仍在运行的子进程，供中断和异常清理使用。'''

    for worker in workers:
        if worker.poll() is None:
            worker.terminate()


def run_workers(args: argparse.Namespace) -> int:
    '''生成分片并为每个非空分片启动独立 GPU 进程。'''

    gpu_ids = parse_gpu_ids(args.gpus)
    if not args.config.resolve().is_file():
        raise FileNotFoundError(f"Config file not found: {args.config.resolve()}")
    images = select_images(
        args.image_dir.resolve(), args.pattern, args.start_index, args.limit
    )
    if args.smoke_test:
        images = images[:1]
        gpu_ids = gpu_ids[:1]
    if not images:
        print("No images selected; nothing to run.", flush=True)
        return 0

    output_dir = args.output_dir.resolve()
    final_mask_dir = (
        args.final_mask_dir.resolve()
        if args.final_mask_dir is not None
        else output_dir / "final_masks"
    )
    shard_dir = output_dir / "shards"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shards = split_images(images, len(gpu_ids))
    commands: list[tuple[str, list[str], Path]] = []
    for gpu_id, shard in zip(gpu_ids, shards):
        if not shard:
            continue
        safe_gpu_id = gpu_id.replace("/", "_").replace(",", "_")
        shard_path = shard_dir / f"gpu_{safe_gpu_id}.txt"
        summary_path = output_dir / f"run_summary_gpu_{safe_gpu_id}.json"
        log_path = log_dir / f"gpu_{safe_gpu_id}.log"
        write_shard_list(shard_path, shard)
        command = build_worker_command(
            args, gpu_id, shard_path, summary_path, final_mask_dir
        )
        commands.append((gpu_id, command, log_path))
        print(
            f"GPU {gpu_id}: {len(shard)} images, list={shard_path}, log={log_path}",
            flush=True,
        )
        if args.dry_run:
            print("  " + " ".join(command), flush=True)

    if args.dry_run:
        return 0

    workers: list[subprocess.Popen] = []
    log_handles = []
    previous_handlers = {}

    def handle_signal(signum, _frame):
        '''收到终端信号时将其转化为 worker 清理和标准退出。'''

        terminate_workers(workers)
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        for gpu_id, command, log_path in commands:
            log_handle = log_path.open("a", encoding="utf-8")
            log_handles.append(log_handle)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            worker = subprocess.Popen(
                command,
                cwd=ENTRY_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            workers.append(worker)
            print(f"Started GPU {gpu_id} worker (pid={worker.pid}).", flush=True)

        return_codes = [worker.wait() for worker in workers]
    except KeyboardInterrupt:
        terminate_workers(workers)
        for worker in workers:
            worker.wait()
        return 130
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for log_handle in log_handles:
            log_handle.close()

    failed = [
        gpu_id
        for (gpu_id, _command, _log_path), code in zip(commands, return_codes)
        if code != 0
    ]
    if failed:
        print(f"Workers failed on GPUs: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"All workers finished. Results: {output_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    '''解析参数并运行多 GPU 调度器。'''

    args = build_parser().parse_args(argv)
    try:
        return run_workers(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

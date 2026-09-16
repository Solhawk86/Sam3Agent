import time
import traceback
from pathlib import Path

from sam3_agent.config.schema import RunConfig

from .dependencies import build_runner
from .images import iter_images, load_image_list
from .prompts import extract_prompt
from .summary import build_summary_path, update_summary, write_error_file


def resolve_images(config: RunConfig) -> tuple[list[Path], Path]:
    """根据 image_list 或 image_dir 配置解析本次要处理的图片。"""
    if config.input.image_list is not None:
        return (
            load_image_list(
                config.input.image_list,
                config.workspace_root,
                config.input.start_index,
                config.input.limit,
            ),
            config.input.image_list,
        )

    if config.input.image_dir is None:
        raise ValueError("Missing config: input.image_dir or input.image_list")
    return (
        iter_images(
            config.input.image_dir,
            config.input.pattern,
            config.input.start_index,
            config.input.limit,
        ),
        config.input.image_dir,
    )


def run_batch(config: RunConfig) -> None:
    """批量运行入口：枚举图片、逐张执行 agent 推理并记录摘要。"""
    image_paths, image_source = resolve_images(config)

    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    if config.output.final_mask_dir is not None:
        config.output.final_mask_dir.mkdir(parents=True, exist_ok=True)
    summary_path = build_summary_path(config.output.output_dir)

    total = len(image_paths)
    print(f"Loaded agent config: {config.config_path}", flush=True)
    print(f"Found {total} images in {image_source}", flush=True)
    if total == 0:
        return

    llm_config, request, sam_tool, get_result_dir, run_single_image_inference = (
        build_runner(config, verbose=config.agent.verbose)
    )

    for idx, image_path in enumerate(image_paths, start=1):
        progress = idx / total * 100
        image_path_resolved = str(Path(image_path).resolve())
        result_dir = get_result_dir(str(config.output.output_dir), image_path_resolved)
        llm_request_counter = {"count": 0}

        def counted_request(*call_args, **call_kwargs):
            """包装 LLM 请求函数，用于统计单张图片的请求次数。"""
            llm_request_counter["count"] += 1
            return request(*call_args, **call_kwargs)

        start_time = time.perf_counter()
        try:
            prompt = extract_prompt(Path(image_path), config.prompt)
            print(
                f"[{idx}/{total} | {progress:6.2f}%] Start {Path(image_path).name} -> prompt='{prompt}'",
                flush=True,
            )
            run_result = run_single_image_inference(
                image_path=image_path_resolved,
                text_prompt=prompt,
                llm_config=llm_config,
                send_generate_request=counted_request,
                segmentation_tool=sam_tool,
                output_dir=str(config.output.output_dir),
                debug=config.agent.debug,
                max_generations=config.agent.max_generations,
                max_box_tasks_per_round=config.agent.max_box_tasks_per_round,
                final_mask_output_dir=(
                    str(config.output.final_mask_dir)
                    if config.output.final_mask_dir is not None
                    else None
                ),
                verbose=config.agent.verbose,
            )
            status = run_result["status"]
            elapsed_sec = time.perf_counter() - start_time
            update_summary(
                summary_path,
                Path(image_path).name,
                status in {"success", "skipped"},
                llm_request_counter["count"],
                elapsed_sec,
                status=status,
                statistics=run_result.get("statistics"),
            )
            print(
                f"[{idx}/{total} | {progress:6.2f}%] {status.upper():7s} {Path(image_path).name}",
                flush=True,
            )
        except Exception:
            elapsed_sec = time.perf_counter() - start_time
            error_text = traceback.format_exc()
            write_error_file(result_dir, error_text)
            update_summary(
                summary_path,
                Path(image_path).name,
                False,
                llm_request_counter["count"],
                elapsed_sec,
                status="error",
            )
            print(
                f"[{idx}/{total} | {progress:6.2f}%] FAILED  {Path(image_path).name}",
                flush=True,
            )

"""Command-line entry point for single-image agent inference."""

import argparse
import os
from pathlib import Path

from .inference import run_single_image_inference
from .llm_client import send_generate_request
from .tools import Sam3Tool


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default="agent_output")
    parser.add_argument("--final-mask-dir", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--bpe-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-generations", type=int, default=20)
    parser.add_argument("--max-box-tasks-per-round", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = args.model or os.environ.get("SAM3_AGENT_MODEL")
    base_url = args.base_url or os.environ.get("SAM3_AGENT_BASE_URL")
    api_key = args.api_key or os.environ.get("SAM3_AGENT_API_KEY")
    if not model:
        raise SystemExit("Set --model or SAM3_AGENT_MODEL")
    if not base_url:
        raise SystemExit("Set --base-url or SAM3_AGENT_BASE_URL")
    if not api_key:
        raise SystemExit("Set --api-key or SAM3_AGENT_API_KEY")

    llm_config = {
        "provider": "openai",
        "name": model,
        "model": model,
        "base_url": base_url,
    }
    sam_tool = Sam3Tool(
        checkpoint_path=args.sam3_checkpoint,
        device=args.sam3_device,
        confidence_threshold=args.confidence_threshold,
        bpe_path=args.bpe_path,
        enable_inst_interactivity=True,
    )
    request = lambda messages, **request_options: send_generate_request(
        messages,
        server_url=base_url,
        model=model,
        api_key=api_key,
        verbose=args.verbose,
        **request_options,
    )
    result = run_single_image_inference(
        image_path=str(Path(args.image).resolve()),
        text_prompt=args.prompt,
        llm_config=llm_config,
        send_generate_request=request,
        segmentation_tool=sam_tool,
        output_dir=args.output_dir,
        debug=args.debug,
        max_generations=args.max_generations,
        max_box_tasks_per_round=args.max_box_tasks_per_round,
        final_mask_output_dir=args.final_mask_dir,
        verbose=args.verbose,
    )
    print(result["output_json_path"])


if __name__ == "__main__":
    main()

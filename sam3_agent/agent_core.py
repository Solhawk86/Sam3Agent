# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

'''使用原生 function calling 驱动四个视觉 grounding 工具。'''

import copy
import json
import os
from typing import Any

from .llm_client import LLMResponse, send_generate_request
from .tools import (
    ToolContext,
    build_agent_tool_registry,
)
from .tools.protocol import SegmentationTool


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    '''在启用 debug 时保存当前消息历史。'''

    if debug and debug_jsonl_path:
        os.makedirs(debug_folder_path, exist_ok=True)
        with open(debug_jsonl_path, "w", encoding="utf-8") as handle:
            for message in messages_list:
                handle.write(json.dumps(message, indent=4, ensure_ascii=False) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    '''成功结束后清理临时 debug 历史文件。'''

    if debug and debug_folder_path:
        try:
            if os.path.exists(debug_jsonl_path):
                os.remove(debug_jsonl_path)
            if os.path.exists(debug_folder_path):
                os.rmdir(debug_folder_path)
        except Exception as exc:
            print(f"Warning: Could not clean up debug files: {exc}")


def save_raw_llm_output(response: Any, output_dir, filename):
    '''保存 LLM 文本和结构化 tool calls，便于排查协议问题。'''

    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    if response is None:
        serialized = ""
    elif isinstance(response, LLMResponse):
        serialized = json.dumps(response.as_dict(), indent=2, ensure_ascii=False)
    elif isinstance(response, str):
        serialized = response
    else:
        serialized = json.dumps(response, indent=2, ensure_ascii=False, default=str)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def count_images(messages):
    '''统计内部消息历史中引用的本地图片数量。'''

    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        total += sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        )
    return total


def _message_has_tool_call(message, tool_name, call_id=None):
    '''判断 assistant 消息是否包含指定的原生 function call。'''

    if message.get("role") != "assistant":
        return False
    for tool_call in message.get("tool_calls", []):
        function = tool_call.get("function", {})
        if function.get("name") != tool_name:
            continue
        if call_id is None or tool_call.get("id") == call_id:
            return True
    return False


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
    latest_segment_call_id=None,
):
    '''保留初始输入和最近一次成功分割以来的原生工具消息。'''

    part1 = copy.deepcopy(messages_list[:2])
    part2_start_idx = None
    for index in range(len(messages_list) - 1, 1, -1):
        if _message_has_tool_call(
            messages_list[index],
            "segment_phrase",
            latest_segment_call_id,
        ):
            part2_start_idx = index
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []
    previously_used = (
        [prompt for prompt in used_text_prompts if prompt != latest_sam3_text_prompt]
        if latest_sam3_text_prompt
        else list(used_text_prompts)
    )
    if part2 and previously_used:
        warning_text = (
            "The following text_prompt values were already tried and must not be "
            f"used again: {sorted(previously_used)}."
        )
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": (
                        "The above image is the raw input image. The initial user "
                        f"input query is: {initial_text_prompt!r}. {warning_text}"
                    ),
                },
            ],
        }

    return [*part1, *part2]


def _remove_previous_result_images(messages):
    '''在逐 mask 检查后移除已被新结果替代的旧渲染图消息。'''

    retained = messages[:2]
    for message in messages[2:]:
        content = message.get("content")
        has_image = isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") == "image"
            for item in content
        )
        if message.get("role") == "user" and has_image:
            continue
        retained.append(message)
    messages[:] = retained


def _parse_single_tool_call(response: LLMResponse):
    '''验证模型恰好返回一个原生工具调用并解析其 JSON 参数。'''

    if not isinstance(response, LLMResponse):
        raise TypeError(
            "send_generate_request must return LLMResponse for agent tool calls"
        )
    if len(response.tool_calls) != 1:
        raise ValueError(
            "Expected exactly one native tool call, got "
            f"{len(response.tool_calls)}"
        )
    tool_call = response.tool_calls[0]
    try:
        arguments = json.loads(tool_call.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON arguments for tool {tool_call.name!r}: "
            f"{tool_call.arguments}"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for tool {tool_call.name!r} must be an object")
    return tool_call, arguments


def _save_failed_history(messages, error_save_dir, img_path, verbose):
    '''在 LLM 无响应时保存导致失败的消息历史。'''

    error_save_path = os.path.join(
        error_save_dir,
        f"{os.path.splitext(os.path.basename(img_path))[0]}_error_history.json",
    )
    with open(error_save_path, "w", encoding="utf-8") as handle:
        json.dump(messages, handle, indent=4, ensure_ascii=False)
    if verbose:
        print("Saved messages history that caused error to:", error_save_path)


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    segmentation_tool: SegmentationTool = None,
    max_generations: int = 20,
    output_dir="agent_output",
    llm_log_dir=None,
    verbose: bool = True,
):
    '''迭代调用多模态模型和四个原生工具完成单图 grounding。'''

    if segmentation_tool is None:
        raise ValueError("segmentation_tool is required")

    sam_output_dir = os.path.join(output_dir, "sam_out")
    error_save_dir = os.path.join(output_dir, "none_out")
    debug_save_dir = os.path.join(output_dir, "agent_debug_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    os.makedirs(error_save_dir, exist_ok=True)
    os.makedirs(debug_save_dir, exist_ok=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(
        current_dir, "system_prompts/system_prompt.txt"
    )
    iterative_prompt_path = os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking.txt"
    )
    with open(system_prompt_path, "r", encoding="utf-8") as handle:
        system_prompt = handle.read().strip()
    with open(iterative_prompt_path, "r", encoding="utf-8") as handle:
        iterative_system_prompt = handle.read().strip()

    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        image_stem = os.path.splitext(os.path.basename(img_path))[0]
        debug_folder_path = os.path.join(debug_save_dir, image_stem)
        debug_jsonl_path = os.path.join(debug_folder_path, "debug_history.json")
        os.makedirs(debug_folder_path, exist_ok=True)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": (
                        "The above image is the raw input image. The initial user "
                        f"input query is: {initial_text_prompt!r}."
                    ),
                },
            ],
        },
    ]
    context = ToolContext(
        image_path=img_path,
        initial_text_prompt=initial_text_prompt,
        sam_output_dir=sam_output_dir,
        iterative_system_prompt=iterative_system_prompt,
        send_generate_request=send_generate_request,
        save_llm_output=lambda response, filename: save_raw_llm_output(
            response, llm_log_dir, filename
        ),
        verbose=verbose,
    )
    registry = build_agent_tool_registry(segmentation_tool)
    generation_count = 0

    while generation_count < max_generations:
        if verbose:
            print(f"> Text prompt: {initial_text_prompt}")
            print(f"> Image path: {img_path}")
            print(f"\n{'-' * 30} Round {generation_count + 1}{'-' * 30}\n")

        response = send_generate_request(
            messages,
            tools=registry.definitions(context),
            tool_choice="required",
            parallel_tool_calls=False,
        )
        save_raw_llm_output(
            response,
            llm_log_dir,
            f"raw_llm_round_{generation_count + 1:03d}.txt",
        )
        if response is None:
            _save_failed_history(messages, error_save_dir, img_path, verbose)
            raise ValueError(
                "Generated response is None. Check the Qwen server and request "
                f"parameters for image path: {img_path}"
            )

        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        tool_call, arguments = _parse_single_tool_call(response)
        messages.append(response.as_assistant_message())
        result = registry.execute(tool_call.name, context, arguments)
        if tool_call.name == "segment_phrase" and result.success:
            context.latest_segment_call_id = tool_call.id

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.as_tool_content(),
            }
        )

        if result.terminal:
            if result.final_outputs is None or result.rendered_image is None:
                raise ValueError(
                    f"Terminal tool {tool_call.name!r} returned incomplete outputs"
                )
            cleanup_debug_files(
                debug,
                debug_folder_path,
                debug_jsonl_path,
            )
            return messages, result.final_outputs, result.rendered_image

        if tool_call.name == "examine_each_mask":
            _remove_previous_result_images(messages)
        if result.image_path is not None:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": result.content.get("message", "Tool result image:"),
                        },
                        {"type": "image", "image": result.image_path},
                    ],
                }
            )

        messages = _prune_messages_for_next_round(
            messages,
            context.used_text_prompts,
            context.latest_text_prompt,
            img_path,
            initial_text_prompt,
            context.latest_segment_call_id,
        )
        if count_images(messages) > 2:
            raise ValueError("Agent history contains more than two images")
        generation_count += 1

    raise ValueError(
        f"Exceeded maximum number of allowed generation requests ({max_generations})"
    )

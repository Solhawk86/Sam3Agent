# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI


@dataclass(frozen=True)
class FunctionToolCall:
    '''保存一个规范化后的原生 function tool call。'''

    id: str
    name: str
    arguments: str

    def as_dict(self) -> Dict[str, Any]:
        '''转换为可追加到 assistant history 的消息字段。'''

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class LLMResponse:
    '''统一表示一次 LLM 返回的文本和原生工具调用。'''

    content: Optional[str]
    tool_calls: tuple[FunctionToolCall, ...] = ()

    def as_assistant_message(self) -> Dict[str, Any]:
        '''转换为 OpenAI-compatible assistant history 消息。'''

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [call.as_dict() for call in self.tool_calls]
        return message

    def as_dict(self) -> Dict[str, Any]:
        '''转换为适合日志持久化的普通字典。'''

        return {
            "content": self.content,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
        }


def _normalize_tool_calls(
    tool_calls: Optional[Iterable[Any]],
) -> tuple[FunctionToolCall, ...]:
    '''把 OpenAI SDK tool call 对象转换为稳定的内部类型。'''

    normalized = []
    for call in tool_calls or []:
        arguments = call.function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        normalized.append(
            FunctionToolCall(
                id=call.id,
                name=call.function.name,
                arguments=arguments,
            )
        )
    return tuple(normalized)


def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=4096,
    verbose=True,
    tools=None,
    tool_choice=None,
    parallel_tool_calls=None,
    extra_body=None,
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)

    Returns:
        LLMResponse: The generated text and native tool calls from the server.
    """
    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]

                    if verbose:
                        print("image_path", image_path)
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path
                        )
                        if base64_image is None:
                            if verbose:
                                print(
                                    f"Warning: Could not convert image to base64: {new_image_path}"
                                )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        )

                    except FileNotFoundError:
                        if verbose:
                            print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        if verbose:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key, base_url=server_url)

    try:
        if verbose:
            print(f"🔍 Calling model {model}...")
        request_kwargs = {
            "model": model,
            "messages": processed_messages,
            "max_completion_tokens": max_tokens,
            "n": 1,
        }
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            request_kwargs["parallel_tool_calls"] = parallel_tool_calls
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(
            **request_kwargs,
        )
        # print(f"Received response: {response.choices[0].message}")

        # Extract the response content
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            return LLMResponse(
                content=message.content,
                tool_calls=_normalize_tool_calls(message.tool_calls),
            )
        else:
            if verbose:
                print(f"Unexpected response format: {response}")
            return None

    except Exception as e:
        if verbose:
            print(f"Request failed: {e}")
        return None


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None

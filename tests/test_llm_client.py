from types import SimpleNamespace

from sam3_agent import llm_client


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="segment_phrase",
                arguments='{"text_prompt":"object"}',
            ),
        )
        message = SimpleNamespace(content="analysis", tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_send_generate_request_passes_and_normalizes_native_tools(monkeypatch):
    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    monkeypatch.setattr(llm_client, "OpenAI", lambda **kwargs: client)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "segment_phrase",
                "description": "segment",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = llm_client.send_generate_request(
        [{"role": "user", "content": "find it"}],
        server_url="http://example.test/v1",
        model="fake-model",
        api_key="fake-key",
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        verbose=False,
    )

    assert completions.kwargs["tools"] == tools
    assert completions.kwargs["tool_choice"] == "required"
    assert completions.kwargs["parallel_tool_calls"] is False
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert response.content == "analysis"
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "segment_phrase"


def test_send_generate_request_preserves_all_images_in_one_call(tmp_path, monkeypatch):
    '''同一请求中的原图、概览和四页局部图均按顺序编码且只调用一次接口。'''

    import base64

    from PIL import Image

    completions = FakeCompletions()
    calls = []
    original_create = completions.create

    def capture_create(**kwargs):
        '''记录实际完成请求次数。'''

        calls.append(kwargs)
        return original_create(**kwargs)

    monkeypatch.setattr(completions, "create", capture_create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_client, "OpenAI", lambda **kwargs: client)
    paths = []
    for index in range(6):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(path)
        paths.append(path)
    content = []
    for index, path in enumerate(paths[1:]):
        content.extend([
            {"type": "text", "text": f"view {index}"},
            {"type": "image", "image": str(path)},
        ])
    llm_client.send_generate_request(
        [
            {"role": "user", "content": [{"type": "image", "image": str(paths[0])}]},
            {"role": "user", "content": content},
        ],
        server_url="http://example.test/v1", model="fake", api_key="fake", verbose=False,
    )
    assert len(calls) == 1
    parts = [part for message in calls[0]["messages"] for part in message["content"]]
    urls = [part["image_url"] for part in parts if part["type"] == "image_url"]
    assert len(urls) == len(paths)
    for url, path in zip(urls, paths):
        assert url["detail"] == "high"
        assert base64.b64decode(url["url"].split(",", 1)[1]) == path.read_bytes()
    assert [part["text"] for part in parts if part["type"] == "text"] == [
        f"view {index}" for index in range(5)
    ]

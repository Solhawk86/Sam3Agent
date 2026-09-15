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

'''使用假 LLM 和 SAM 验证完整批量记忆流程。'''

import copy
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import yaml
from PIL import Image

from sam3_agent.batch import runner as batch_runner
from sam3_agent.config.factory import build_run_config
from sam3_agent.inference import (
    _build_union_binary_mask,
    get_result_dir,
    run_single_image_inference,
)
from sam3_agent.llm_client import FunctionToolCall, LLMResponse
from sam3_agent.tools.protocol import SegmentationResult


def decision(
    text=None, boxes=None, accept=(), reject=(), replace=(), inspect=(), finish=None
):
    '''构造完整的复合工具参数，审核理由使用固定测试结论。'''

    return {
        "review": {
            "accept": [{"mask_id": key, "reason": "correct target"} for key in accept],
            "reject": [{"mask_id": key, "reason": "wrong region"} for key in reject],
            "replace": list(replace),
        },
        "text_prompt": text,
        "boxes": [{"box": box} for box in (boxes or [])],
        "inspect_mask_ids": list(inspect),
        "finish": finish is not None,
        "finish_reason": finish,
    }


def response(arguments, call_id="call", name="advance_segmentation"):
    '''构造 SDK 无关的原生工具响应。'''

    return LLMResponse(
        None,
        (
            FunctionToolCall(
                call_id,
                name,
                arguments if isinstance(arguments, str) else json.dumps(arguments),
            ),
        ),
    )


def rectangle(left=2, top=2, right=8, bottom=8):
    '''生成可验证并集和替换结果的二值矩形。'''

    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


class FakeBackend:
    '''记录任务顺序并按脚本返回候选或模拟错误。'''

    def __init__(self, outcomes=None):
        '''初始化不依赖 GPU 的执行记录。'''

        self.outcomes = list(outcomes or [])
        self.calls = []
        self.prepared = False
        self.builds = 0

    def prepare(self):
        '''模拟幂等模型准备。'''

        if not self.prepared:
            self.prepared = True
            self.builds += 1

    def _segment(self, branch, arguments, image_path, output_dir):
        '''在独立目录保存本次候选，不接触主运行记忆。'''

        assert self.prepared
        self.calls.append((branch, arguments, output_dir))
        outcome = self.outcomes.pop(0) if self.outcomes else rectangle()
        if isinstance(outcome, Exception):
            raise outcome
        masks = [] if outcome is None else (outcome if isinstance(outcome, list) else [outcome])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded = [
            mask_utils.encode(np.asfortranarray(mask))["counts"].decode()
            for mask in masks
        ]
        data = {
            "original_image_path": image_path,
            "orig_img_h": 10,
            "orig_img_w": 12,
            "pred_boxes": [[0, 0, 1, 1] for _ in masks],
            "pred_scores": [0.9 for _ in masks],
            "pred_masks": encoded,
        }
        output_json = output_dir / "result.json"
        output_image = output_dir / "result.png"
        output_json.write_text(json.dumps(data))
        with Image.open(image_path) as image:
            image.save(output_image)
        return SegmentationResult(str(output_json), str(output_image), data)

    def segment_phrase(self, image_path, text_prompt, output_dir, verbose=False):
        '''记录文本分支任务。'''

        return self._segment("text", text_prompt, image_path, output_dir)

    def segment_instance_with_box(self, image_path, box, output_dir, verbose=False):
        '''记录框分支任务。'''

        return self._segment("box", box, image_path, output_dir)


class FakeRequest:
    '''保存每次真实主循环请求，禁止额外隐藏请求。'''

    def __init__(self, responses, backend):
        '''绑定已加载后端和有限响应序列。'''

        self.responses = iter(responses)
        self.backend = backend
        self.calls = []

    def __call__(self, messages, **options):
        '''确保先准备后端，再消耗一条模型响应。'''

        assert self.backend.prepared
        self.calls.append((copy.deepcopy(messages), options))
        return next(self.responses)


@pytest.fixture
def run_agent(tmp_path):
    '''提供带隔离输入、输出目录的完整推理入口。'''

    image_path = tmp_path / "input.png"
    Image.new("RGB", (12, 10), "gray").save(image_path)

    def run(responses, backend=None, **kwargs):
        '''使用脚本响应执行单图推理并读取状态快照。'''

        backend = backend or FakeBackend()
        request = FakeRequest(responses, backend)
        result = run_single_image_inference(
            str(image_path),
            "fish",
            {"name": "fake"},
            request,
            backend,
            output_dir=str(tmp_path / "output"),
            final_mask_output_dir=str(tmp_path / "final"),
            **kwargs,
        )
        state = json.loads(
            (Path(result["result_dir"]) / "memory" / "state.json").read_text()
        )
        return result, state, backend, request

    return run


def test_batch_supplies_clean_initial_category_to_agent(tmp_path, monkeypatch):
    '''完整批处理通过假后端验证首次模型输入及分割请求均使用后端类别词。'''

    image_path = tmp_path / "MAS_MarineFish_GhostPipeFish_Cam_1.jpg"
    Image.new("RGB", (12, 10), "gray").save(image_path)
    config_path = Path(__file__).parents[1] / "agent_config.yaml"
    config = build_run_config(
        {
            "input": {"image_dir": str(tmp_path)},
            "output": {"output_dir": str(tmp_path / "output")},
            "prompt": yaml.safe_load(config_path.read_text())["prompt"],
            "agent": {"max_generations": 2},
        },
        config_path,
        tmp_path,
    )
    backend = FakeBackend()
    request = FakeRequest(
        [
            response(decision(text="ghost pipe fish"), "first"),
            response(decision(accept=["m1"], finish="complete"), "last"),
        ],
        backend,
    )

    def build_fake_runner(config, verbose=False):
        '''复用真实单图入口，只替换模型和服务依赖。'''

        return (
            {"name": "fake"},
            request,
            backend,
            get_result_dir,
            run_single_image_inference,
        )

    monkeypatch.setattr(batch_runner, "build_runner", build_fake_runner)
    batch_runner.run_batch(config)

    assert len(request.calls) == 2
    for messages, _ in request.calls:
        summary = json.loads(messages[1]["content"][1]["text"])
        assert summary["query"] == "ghost pipe fish"
    assert [call[:2] for call in backend.calls] == [("text", "ghost pipe fish")]
    result_dir = Path(get_result_dir(str(config.output.output_dir), str(image_path)))
    prediction = json.loads((result_dir / "pred.json").read_text())
    assert prediction["text_prompt"] == "ghost pipe fish"
    assert prediction["status"] == "success"


def test_text_and_three_boxes_finish_in_two_requests(run_agent):
    '''一个文本与三个框只需两次 LLM 请求，跨分支相同像素仍独立入库。'''

    result, state, backend, request = run_agent(
        [
            response(
                decision(
                    text=" fish ", boxes=[[0, 0, 3, 4], [3, 0, 6, 4], [6, 0, 9, 4]]
                ),
                "first",
            ),
            response(
                decision(accept=["m1", "m2", "m3", "m4"], finish="complete"), "last"
            ),
        ]
    )
    assert result["status"] == "success"
    assert [call[0] for call in backend.calls] == ["text", "box", "box", "box"]
    assert len(request.calls) == result["statistics"]["llm_requests"] == 2
    assert len(state["candidates"]) == 4
    assert len({item["rle"] for item in state["candidates"].values()}) == 1
    assert len({call[2] for call in backend.calls}) == 4
    for round_index, (messages, options) in enumerate(request.calls):
        assert [item["function"]["name"] for item in options["tools"]] == [
            "advance_segmentation"
        ]
        assert options["parallel_tool_calls"] is False
        assert (
            sum(
                part.get("type") == "image"
                for message in messages
                for part in (message.get("content") or [])
                if isinstance(part, dict)
            )
            == (1 if round_index == 0 else 3)
        )
    assert Path(result["final_mask_path"]).is_file()
    assert set(np.asarray(Image.open(result["final_mask_path"])).ravel()) == {0, 255}


@pytest.mark.parametrize("branch", ["text", "box"])
def test_masks_and_closeups_arrive_together_without_inspection_turn(run_agent, branch):
    '''文本和框分割均在首次结果回传时附带局部图，两次请求即可审核结束。'''

    first = decision(text="fish") if branch == "text" else decision(boxes=[[1, 1, 9, 9]])
    result, state, backend, request = run_agent(
        [
            response(first, "segment"),
            response(decision(accept=["m1"], finish="complete"), "review"),
        ],
        max_generations=2,
    )

    assert result["status"] == "success"
    assert len(request.calls) == result["statistics"]["llm_requests"] == 2
    assert len(backend.calls) == 1
    messages = request.calls[1][0]
    summary = json.loads(messages[1]["content"][1]["text"])
    tool_result = json.loads(messages[3]["content"])
    assert summary["inspection_mask_ids"] == ["m1"]
    assert tool_result["inspection_mask_ids"] == ["m1"]
    assert summary["candidates"][0]["status"] == "pending"
    assert len(image_paths(messages)) == 3
    assert "m1" in messages[-1]["content"][2]["text"]
    with Image.open(messages[-1]["content"][3]["image"]) as page:
        assert page.width >= 1024 and page.height >= 1024
    assert state["candidates"]["m1"]["status"] == "accepted"
    assert state["inspection_ids"] == []


def test_empty_segmentation_does_not_add_closeups(run_agent):
    '''无 mask 时不生成虚假局部图，仍允许下一轮判断无目标。'''

    result, _, _, request = run_agent(
        [response(decision(text="fish")), response(decision(finish="no_target"))],
        FakeBackend([None]),
    )
    assert result["status"] == "success"
    summary = json.loads(request.calls[1][0][1]["content"][1]["text"])
    assert summary["inspection_mask_ids"] == []
    assert summary["candidates"] == []


def test_review_and_next_batch_preserve_then_replace(run_agent):
    '''下一轮审核和新任务同时执行，替换旧结果时保留其他目标共享像素。'''

    first, other, better = rectangle(), rectangle(6, 1, 11, 5), rectangle(2, 2, 5, 6)
    result, state, _, request = run_agent(
        [
            response(decision(text="fish", boxes=[[6, 1, 11, 5]])),
            response(decision(accept=["m1", "m2"], boxes=[[2, 2, 5, 6]])),
            response(
                decision(
                    replace=[
                        {
                            "old_mask_ids": ["m1"],
                            "new_mask_ids": ["m3"],
                            "reason": "less background",
                        }
                    ],
                    finish="complete",
                )
            ),
        ],
        FakeBackend([first, other, better]),
    )
    assert len(request.calls) == 3
    assert state["candidates"]["m1"]["status"] == "superseded"
    assert state["candidates"]["m2"]["status"] == "accepted"
    outputs = json.loads(Path(result["output_json_path"]).read_text())
    np.testing.assert_array_equal(
        _build_union_binary_mask(outputs), (other | better) * 255
    )


@pytest.mark.parametrize(
    "bad_change",
    [
        {"boxes": [{"box": [0, 0, 100, 5]}]},
        {
            "review": {
                "accept": [{"mask_id": "m1", "reason": "ok"}],
                "reject": [{"mask_id": "m1", "reason": "wrong"}],
                "replace": [],
            }
        },
        {
            "review": {
                "accept": [{"mask_id": "m999", "reason": "future"}],
                "reject": [],
                "replace": [],
            }
        },
        {"finish": True, "finish_reason": "complete"},
        {"finish": 1},
    ],
)
def test_invalid_batch_does_not_apply_reviews_or_execute(run_agent, bad_change):
    '''整批无效时不产生部分审核，也不执行其文本任务。'''

    invalid = decision(accept=["m1"], text="another fish")
    invalid.update(bad_change)
    result, state, backend, request = run_agent(
        [
            response(decision(text="fish")),
            response(invalid),
        ],
        max_generations=2,
    )
    assert result["status"] == "partial"
    assert len(backend.calls) == 1
    assert state["candidates"]["m1"]["status"] == "pending"
    assert state["review_history"] == []
    assert state["statistics"]["llm_requests"] == 2


def test_exact_requests_reuse_empty_and_nonempty_results(run_agent):
    '''同分支相同参数复用结果，正常空结果也进入请求索引。'''

    result, state, backend, _ = run_agent(
        [
            response(decision(text="fish", boxes=[[0, 0, 4, 4], [0.0, 0, 4, 4]])),
            response(decision(text=" fish ", boxes=[[0, 0, 4, 4]], accept=["m1"])),
            response(decision(finish="complete")),
        ],
        FakeBackend([rectangle(), None]),
    )
    assert result["status"] == "success"
    assert len(backend.calls) == 2
    assert state["statistics"]["cache_hits"] == 3
    assert len(state["candidates"]) == 1


def test_recoverable_error_does_not_clear_success_and_can_retry(run_agent):
    '''单项普通执行错误不阻止其他任务，并允许之后重试相同输入。'''

    result, state, backend, _ = run_agent(
        [
            response(decision(text="fish", boxes=[[0, 0, 4, 4], [5, 0, 9, 4]])),
            response(decision(accept=["m1", "m2"], boxes=[[0, 0, 4, 4]])),
            response(decision(accept=["m3"], finish="complete")),
        ],
        FakeBackend(
            [
                rectangle(),
                RuntimeError("temporary model failure"),
                rectangle(),
                rectangle(),
            ]
        ),
    )
    assert result["status"] == "success"
    assert len(backend.calls) == 4
    assert state["attempts"]["t2"]["status"] == "error"
    assert state["candidates"]["m1"]["status"] == "accepted"


def test_cuda_error_stops_remaining_tasks_and_exports_accepted_only(run_agent):
    '''CUDA 错误停止剩余任务，保留本轮已审核结果为部分输出。'''

    result, state, backend, request = run_agent(
        [
            response(decision(text="fish")),
            response(decision(accept=["m1"], boxes=[[0, 0, 4, 4], [5, 0, 9, 4]])),
        ],
        FakeBackend([rectangle(), RuntimeError("CUDA out of memory")]),
    )
    assert result["status"] == "partial"
    assert result["termination_reason"] == "backend_error"
    assert len(backend.calls) == len(request.calls) == 2
    assert state["attempts"]["t3"]["status"] == "not_run"
    assert (
        len(json.loads(Path(result["output_json_path"]).read_text())["pred_masks"]) == 1
    )
    assert Path(result["final_mask_path"]).name == "partial_mask.png"
    assert not (Path(result["result_dir"]) / "pred.json").exists()


def test_rejected_mask_requires_visible_inspection_before_reaccept(run_agent):
    '''拒绝候选重新接受前先查看局部图，检查动作不触发隐藏请求。'''

    result, state, backend, request = run_agent(
        [
            response(decision(text="fish")),
            response(decision(reject=["m1"])),
            response(decision(accept=["m1"], finish="complete")),
            response(decision(inspect=["m1"])),
            response(decision(accept=["m1"], finish="complete")),
        ]
    )
    assert result["status"] == "success"
    assert len(backend.calls) == 1
    assert len(request.calls) == 5
    assert state["candidates"]["m1"]["status"] == "accepted"
    assert "invalid_decision" in request.calls[3][0][3]["content"]


def test_partial_rerun_keeps_old_artifacts_and_does_not_skip(run_agent):
    '''不完整运行可重新执行，旧运行的候选、历史和部分结果保留。'''

    first, _, backend, _ = run_agent(
        [response(decision(text="fish"))], max_generations=1
    )
    second, _, _, _ = run_agent(
        [
            response(decision(text="fish")),
            response(decision(accept=["m1"], finish="complete")),
        ],
        backend,
    )
    assert first["status"] == "partial" and second["status"] == "success"
    assert first["run_dir"] != second["run_dir"]
    assert (Path(first["run_dir"]) / "partial_pred.json").is_file()
    assert backend.builds == 1


@pytest.mark.parametrize(
    "reply,reason",
    [
        (None, "llm_no_response"),
        (LLMResponse("no tool call"), "protocol_error"),
    ],
)
def test_invalid_response_preserves_audit_and_partial_state(run_agent, reply, reason):
    '''无响应或不可恢复协议错误不伪造成功结论。'''

    result, state, _, request = run_agent([reply])
    assert result["status"] == "partial"
    assert state["termination_reason"] == reason
    assert len(request.calls) == 1


def test_json_error_and_unknown_tool_are_recoverable(run_agent):
    '''可回复到单一调用的协议错误占用预算，但允许下一轮纠正。'''

    result, _, backend, request = run_agent(
        [
            response("{broken"),
            response({}, name="segment_phrase"),
            response(decision(finish="no_target")),
        ]
    )
    assert result["status"] == "success"
    assert result["termination_reason"] == "no_target"
    assert len(request.calls) == 3 and not backend.calls


def test_zero_budget_exports_empty_partial_without_llm(run_agent):
    '''零预算不发起 LLM 请求，也不自动认定无目标。'''

    result, state, backend, request = run_agent([], max_generations=0)
    assert result["status"] == "partial"
    assert state["termination_reason"] == "budget_exhausted"
    assert not backend.calls and not request.calls


def test_startup_failure_never_calls_llm(run_agent):
    '''模型加载失败在任何 LLM 请求之前报错并保存启动状态。'''

    class BrokenBackend(FakeBackend):
        '''模拟模型启动失败。'''

        def prepare(self):
            '''直接报告缺少模型权重。'''

            raise RuntimeError("missing checkpoint")

    with pytest.raises(RuntimeError, match="missing checkpoint"):
        run_agent([], BrokenBackend())


def image_paths(messages):
    '''按发送顺序提取请求里的全部图片路径。'''

    return [
        part["image"] for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]


@pytest.mark.parametrize("count", [0, 1, 4, 5, 13])
def test_all_pages_arrive_in_one_review_request(run_agent, count):
    '''多个候选分页仍只需一次分割请求和一次审核请求。'''

    ids = [f"m{index}" for index in range(1, count + 1)]
    result, _, _, request = run_agent(
        [
            response(decision(text="fish"), "segment"),
            response(decision(accept=ids, finish="complete" if count else "no_target"), "review"),
        ],
        FakeBackend([[rectangle() for _ in ids]]),
    )
    assert result["status"] == "success"
    assert len(request.calls) == result["statistics"]["llm_requests"] == 2
    assert len(image_paths(request.calls[0][0])) == 1
    paths = image_paths(request.calls[1][0])
    assert len(paths) == 2 + (count + 3) // 4
    assert Path(paths[1]).name == "round_001_overview.png"
    assert [Path(path).name for path in paths[2:]] == [
        f"round_001_closeups_{index:03d}.png" for index in range(1, (count + 3) // 4 + 1)
    ]
    assert all(Path(path).is_file() for path in paths)
    saved = json.loads((Path(result["run_dir"]) / "rounds" / "round_002.json").read_text())
    assert image_paths(saved) == paths
    assert not (Path(result["run_dir"]) / "rounds" / "round_001.png").exists()
    assert request.calls[1][0][2]["tool_calls"][0]["id"] == request.calls[1][0][3]["tool_call_id"]


def test_page_groups_replace_and_survive_invalid_decision(run_agent):
    '''分页整体更新，无效审核保留整组图片，审核完成后旧页全部退出请求。'''

    ids = [f"m{index}" for index in range(1, 6)]
    result, _, _, request = run_agent(
        [
            response(decision(text="fish"), "segment"),
            response(decision(accept=ids[:4]), "partial_review"),
            response(decision(accept=["m999"]), "invalid"),
            response(decision(accept=ids[4:]), "review"),
            response(decision(finish="complete"), "finish"),
        ],
        FakeBackend([[rectangle() for _ in ids]]),
    )
    assert result["status"] == "success"
    groups = [image_paths(messages) for messages, _ in request.calls]
    assert [len(group) for group in groups] == [1, 4, 3, 3, 2]
    assert groups[2] == groups[3]
    assert set(groups[1][1:]).isdisjoint(groups[2][1:])
    assert Path(groups[4][1]).name == "round_004_overview.png"
    assert "invalid_decision" in request.calls[3][0][3]["content"]
    assert json.loads(request.calls[3][0][1]["content"][1]["text"])["inspection_mask_ids"] == ["m5"]
    for messages, _ in request.calls[1:]:
        assert messages[2]["tool_calls"][0]["id"] == messages[3]["tool_call_id"]


def test_runner_accepts_legacy_single_image_tool_result(run_agent, monkeypatch):
    '''仅返回旧单图字段的工具仍能反馈一张图片且保持位置参数兼容。'''

    from sam3_agent.tools.advance_segmentation import AdvanceSegmentationTool
    from sam3_agent.tools.protocol import ToolResult

    original = AdvanceSegmentationTool.execute

    def legacy_execute(self, context, arguments):
        '''模拟没有多图字段的旧工具输出。'''

        result = original(self, context, arguments)
        return ToolResult(result.content, result.image_path, result.terminal, result.success)

    monkeypatch.setattr(AdvanceSegmentationTool, "execute", legacy_execute)
    result, _, _, request = run_agent([
        response(decision(text="fish")),
        response(decision(accept=["m1"], finish="complete")),
    ])
    assert result["status"] == "success"
    assert len(image_paths(request.calls[1][0])) == 2

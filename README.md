# SAM3 Agent

An independent visual grounding agent that uses an externally installed SAM3
image model as an in-process segmentation tool.

## Install

Install SAM3 separately, then install this project:

```bash
pip install -e /path/to/sam3
pip install -e .
```

The runtime requires a compatible PyTorch/CUDA environment for the selected
SAM3 checkpoint. The SAM3 repository and checkpoint documentation are the
source of truth for those requirements.

## Run

Configure an OpenAI-compatible multimodal endpoint without putting credentials
in source files:

```bash
export SAM3_AGENT_API_KEY="..."
export SAM3_AGENT_BASE_URL="https://example/v1"
export SAM3_AGENT_MODEL="your-vision-model"
```

Then run one image:

```bash
python -m sam3_agent.cli \
  --image image.jpg \
  --prompt "the person on the left" \
  --output-dir outputs \
  --sam3-checkpoint /path/to/checkpoint.pt
```

The command writes the final JSON, rendered mask image, binary mask, agent
history, and raw LLM responses below the output directory.

## Python API

```python
from sam3_agent.inference import run_single_image_inference
from functools import partial

from sam3_agent.llm_client import send_generate_request
from sam3_agent.tools import Sam3Tool

sam_tool = Sam3Tool(
    checkpoint_path="/path/to/checkpoint.pt",
    enable_inst_interactivity=True,
)
request = partial(
    send_generate_request,
    server_url="https://example/v1",
    model="your-vision-model",
    api_key="...",
)
result = run_single_image_inference(
    image_path="image.jpg",
    text_prompt="the person on the left",
    llm_config={"name": "vision-model"},
    send_generate_request=request,
    segmentation_tool=sam_tool,
    output_dir="outputs",
)
```

The agent passes native OpenAI-compatible function definitions to the model and
expects exactly one `assistant.tool_calls` entry per agent round. The endpoint must
have native tool-call parsing enabled; XML-style `<tool>` responses are not
supported.

For Qwen3 reasoning models served through vLLM, disable thinking while the server
parses function calls by adding this to the YAML `llm` section:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

## Batch decisions and memory

The agent exposes one native tool, `advance_segmentation`. Each response can
review existing masks and submit one text prompt plus up to four pixel-space
`[x1, y1, x2, y2]` boxes. SAM tasks run sequentially on the same resident model;
the next LLM request reviews all their results together. No inner LLM requests
are made for individual masks. `inspect_mask_ids` prepares crops for the next
main request.

The develop branch contains only this workflow. The previous text-only agent
remains on master; there is no mode switch. The seven standalone SAM adapters
remain importable, but only text and box adapters are used by this agent.

```yaml
agent:
  max_generations: 20
  max_box_tasks_per_round: 4
  debug: false
  verbose: false
```

Both entry points accept `--max-box-tasks-per-round`. Python callers can pass
`max_box_tasks_per_round=4` to `run_single_image_inference` or `agent_inference`.
Use `Sam3Tool(enable_inst_interactivity=True)` for the Python API. The CLI enables
this automatically. Model loading and interactive-head validation finish before
the first LLM request. One model instance serves all images in a batch; an empty
input list does not load the model. Existing results in `pred.json` are skipped,
so use a separate output directory when comparing against master.

A complete tool argument object looks like:

```json
{
  "review": {"accept": [], "reject": [], "replace": []},
  "text_prompt": "fish",
  "boxes": [{"box": [100, 120, 350, 380]}],
  "inspect_mask_ids": [],
  "finish": false,
  "finish_reason": null
}
```

`review.accept` and `review.reject` contain objects with `mask_id` and a short
`reason`. Each replacement contains `old_mask_ids`, `new_mask_ids`, and `reason`.
IDs such as `m1` stay stable throughout an image. All review operations and task
arguments are validated before anything executes. New candidates start pending;
only accepted masks enter the final binary union. Explicit replacement removes
old masks from that union. Rejected or superseded masks must be inspected before
reconsideration. Exact requests are reused within their branch; there is no
cross-branch mask matching or IoU-based deduplication.

To finish, set `finish=true` and `finish_reason` to `complete` or `no_target`, with
no new tasks or inspection requests. Final review may accompany this call, but
no pending candidates may remain. `complete` requires accepted masks; `no_target`
requires none. A failed SAM attempt does not establish absence.

Each image writes a current `memory/state.json` and an append-only
`memory/events.jsonl`. Full per-run artifacts live under `memory/runs/<run_id>/`:
state, events, attempts, raw LLM responses, round input JSON and labeled boards.
Retrying an incomplete image preserves its previous run. This is an audit trail,
not automatic checkpoint recovery.

Successful runs write `pred.json`, `pred.png`, `history.json`, and a binary mask
(`pred_mask.png` locally unless a final mask directory was supplied). Budget
exhaustion or a fatal runtime failure exports only accepted masks as
`partial_pred.json`, `partial_pred.png`, `partial_history.json`, and
`partial_mask.png`. Partial runs never create the success marker `pred.json` or
write into the final mask directory, and therefore can be retried. The batch
summary records explicit success/partial/error status, request counts and timing.

Timing fields separate startup, LLM requests, text/box tool calls, board rendering,
and memory storage. Text/box wall times include their original artifact writing
and synchronize CUDA work; they are not GPU-kernel-only measurements. `total_seconds`
measures the agent loop including preparation; the batch summary's elapsed time
also includes final export. No latency or accuracy improvement is asserted without
a real-model evaluation. Binary labels are never passed to the agent.

The implementation specification is in
[the batch segmentation memory plan](docs/batch_segmentation_memory_plan.md).
Automated checks and single-image integration results are recorded in
[the validation notes](docs/batch_memory_validation.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The copied agent implementation retains the original SAM3 copyright and
license notices. See `LICENSE`.

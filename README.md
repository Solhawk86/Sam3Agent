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

sam_tool = Sam3Tool(checkpoint_path="/path/to/checkpoint.pt")
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

The four public agent tools are available as independent classes:

```python
from sam3_agent.tools import (
    ExamineEachMaskTool,
    ReportNoMaskTool,
    SegmentPhraseTool,
    SelectMasksAndReturnTool,
    ToolRegistry,
)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The copied agent implementation retains the original SAM3 copyright and
license notices. See `LICENSE`.

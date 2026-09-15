# Repository Guidelines

## Project Structure & Module Organization

This repository packages an independent SAM3-driven visual grounding agent. Core source lives in `sam3_agent/`: `agent_core.py` contains the iterative agent loop, `inference.py` handles single-image inference and output writing, `llm_client.py` wraps OpenAI-compatible chat APIs, and `tools/` contains SAM3 segmentation adapters. Batch entry logic is split under `sam3_agent/batch/`, while YAML loading and structured runtime config live under `sam3_agent/config/`. Tests are in `tests/`. Runtime configuration for the local batch entry is `agent_config.yaml`; `run_agent.py` is a thin CLI wrapper.

## Build, Test, and Development Commands

Install for development:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

Check syntax after focused edits:

```bash
python -m py_compile run_agent.py sam3_agent/**/*.py
```

Run the package CLI for one image:

```bash
python -m sam3_agent.cli --image image.jpg --prompt "object" --output-dir outputs
```

Run the YAML batch entry without loading models:

```bash
python run_agent.py --limit 0
```

## Coding Style & Naming Conventions

Use Python 3.10+ with 4-space indentation and type hints for new public helpers. Use `snake_case` for functions, variables, and files; use `PascalCase` for dataclasses and tool classes. New functionality must live under `sam3_agent/`, grouped by feature folder and concrete Python file, for example `sam3_agent/config/loader.py` or `sam3_agent/batch/images.py`; do not pile unrelated helpers into one large module.

Before adding a new helper, search existing code first (`rg "function_name|keyword" sam3_agent tests`) and reuse or extend an existing function when it already fits. Avoid creating scattered utility functions with overlapping behavior.

Every newly defined function should include a short Chinese docstring wrapped with triple single quotes:

```python
def extract_prompt(...):
    '''
    按照配置从图片文件名中提取文本提示词。
    '''
```

Keep comments concise and explanatory; prefer docstrings that describe purpose and boundaries, not line-by-line implementation.

## Testing Guidelines

Tests use `pytest`; files should be named `test_*.py` and placed under `tests/`. Prefer fake LLM/SAM3 tools for agent-flow tests so tests do not require CUDA, checkpoints, or external API calls. Add focused tests when changing prompt extraction, config merging, output paths, or agent loop behavior.

## Commit & Pull Request Guidelines

Current history uses concise imperative commits, for example `Create standalone SAM3 agent project`. Keep commit subjects short and action-oriented. Pull requests should describe the behavior change, list test commands run, note any config or checkpoint assumptions, and include sample output paths when changing inference or batch outputs.

## Security & Configuration Tips

Avoid hard-coding real API keys in Python source. Prefer environment variables or local-only YAML overrides for credentials. SAM3 checkpoints and generated masks can be large; do not commit model weights, output folders, cache directories, or private datasets unless explicitly intended.

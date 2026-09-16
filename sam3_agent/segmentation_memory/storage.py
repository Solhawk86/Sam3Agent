'''单图运行产物与原子状态快照。'''

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import SegmentationMemory


def write_json_atomic(path: Path, data: Any) -> None:
    '''先完整写入同目录临时文件，再替换目标 JSON。'''

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class MemorySession:
    '''隔离每次运行的完整产物，并维护便于查看的最新快照。'''

    def __init__(self, result_dir: Path, memory: SegmentationMemory):
        '''创建唯一运行目录，重跑时保留旧审计记录。'''

        self.memory = memory
        self.root = result_dir / "memory"
        self.run_dir = self.root / "runs" / memory.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rounds_dir = self.run_dir / "rounds"
        self.rounds_dir.mkdir()
        self.save_event("run_started", {"run_id": memory.run_id})

    def save(self) -> None:
        '''保存当前运行和最新运行指针，不承诺断点恢复。'''

        start = time.perf_counter()
        snapshot = asdict(self.memory)
        snapshot["run_dir"] = str(self.run_dir)
        write_json_atomic(self.run_dir / "state.json", snapshot)
        write_json_atomic(self.root / "state.json", snapshot)
        self.memory.statistics.storage_seconds += time.perf_counter() - start

    def save_event(self, kind: str, data: dict[str, Any]) -> None:
        '''追加带运行身份的事件并立即保存状态。'''

        start = time.perf_counter()
        event = {
            "run_id": self.memory.run_id,
            "round": self.memory.round_number,
            "kind": kind,
            **data,
        }
        line = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        for path in (self.run_dir / "events.jsonl", self.root / "events.jsonl"):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        self.memory.statistics.storage_seconds += time.perf_counter() - start
        self.save()

    def save_round(self, messages: list[dict[str, Any]]) -> None:
        '''保存本轮实际发送的消息，便于重现图文审核上下文。'''

        start = time.perf_counter()
        path = self.rounds_dir / f"round_{self.memory.round_number:03d}.json"
        write_json_atomic(path, messages)
        self.memory.statistics.storage_seconds += time.perf_counter() - start

    def save_response(self, response: Any) -> None:
        '''保存原始模型响应，不覆盖其他运行的请求日志。'''

        data = response.as_dict() if hasattr(response, "as_dict") else repr(response)
        path = self.run_dir / f"raw_llm_round_{self.memory.round_number:03d}.json"
        write_json_atomic(path, data)

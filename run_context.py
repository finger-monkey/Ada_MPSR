from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


@dataclass
class RunContext:
    enabled: bool = False
    output_dir: Path | str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    _active: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = uuid.uuid4().hex
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    @classmethod
    def disabled(cls) -> "RunContext":
        return cls(enabled=False)

    @classmethod
    def enabled_context(
        cls,
        output_dir: Path | str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RunContext":
        return cls(True, output_dir, run_id, metadata or {})

    def on_start(self, entrypoint: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._active[entrypoint] = {
            "entrypoint": entrypoint,
            "started_at": _timestamp(),
            "metadata": _json_value(metadata or {}),
        }
        self._write()

    def on_end(
        self,
        entrypoint: str,
        status: str = "completed",
        result_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._active.pop(entrypoint, {
            "entrypoint": entrypoint,
            "started_at": None,
            "metadata": {},
        })
        record.update({
            "finished_at": _timestamp(),
            "status": status,
            "result": _json_value(result_metadata or {}),
        })
        self.records.append(record)
        self._write()

    def finish(self, status: str = "completed", metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self.metadata.update(metadata or {})
        self.metadata["status"] = status
        self.metadata["finished_at"] = _timestamp()
        self._write()

    def _write(self) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "started_at": self.metadata.setdefault("started_at", _timestamp()),
            "metadata": _json_value(self.metadata),
            "active": _json_value(self._active),
            "records": _json_value(self.records),
        }
        target = self.output_dir / "run_context.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)


_DEFAULT_CONTEXT = RunContext.disabled()


def get_default_context() -> RunContext:
    return _DEFAULT_CONTEXT


def create_run_context(
    output_dir: Path | str,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunContext:
    return RunContext.enabled_context(output_dir, run_id, metadata)

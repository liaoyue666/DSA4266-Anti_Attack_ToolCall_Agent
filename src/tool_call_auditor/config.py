from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PilotConfig(BaseModel):
    benchmark_version: str = "v1.2.2"
    suite: str = "workspace"
    provider: Literal["agentdojo", "deepseek"] = "agentdojo"
    model: str | None = None
    model_id: str | None = None
    attack: str = "important_instructions"
    decisions: list[Literal["ALLOW", "BLOCK"]] = Field(default_factory=lambda: ["ALLOW", "BLOCK"])
    repetitions: int = Field(default=3, ge=1)
    user_tasks: list[str]
    injection_tasks: list[str]
    consequential_tools: list[str]
    output_dir: Path = Path("runs/workspace_pilot")
    export_dir: Path | None = None
    tool_output_format: Literal["yaml", "json"] = "yaml"
    temperature: float = 0.0

    @model_validator(mode="after")
    def require_unique_entries(self):
        for field in ("decisions", "user_tasks", "injection_tasks", "consequential_tools"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} contains duplicates")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PilotConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

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
    run_mode: Literal["both", "benign_only", "attacked_only"] = "both"
    attacked_task_pairs: list[tuple[str, str]] | None = None
    run_attack_goal_checks: bool = True
    max_episode_attempts: int = Field(default=2, ge=1, le=3)
    source_id: str | None = None
    output_dir: Path = Path("runs/workspace_pilot")
    export_dir: Path | None = None
    tool_output_format: Literal["yaml", "json"] = "yaml"
    temperature: float = 0.0

    @model_validator(mode="after")
    def require_unique_entries(self):
        for field in (
            "decisions",
            "user_tasks",
            "injection_tasks",
            "consequential_tools",
        ):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} contains duplicates")
        if not self.user_tasks:
            raise ValueError("user_tasks must name at least one task explicitly")
        if not self.decisions:
            raise ValueError("decisions must name at least one policy explicitly")
        if not self.consequential_tools:
            raise ValueError("consequential_tools must name at least one tool explicitly")
        if self.run_mode == "benign_only" and self.injection_tasks:
            raise ValueError("benign_only requires injection_tasks: []")
        if self.run_mode == "attacked_only" and not self.injection_tasks:
            raise ValueError("attacked_only requires explicit injection_tasks")
        if self.attacked_task_pairs is not None:
            if len(self.attacked_task_pairs) != len(set(self.attacked_task_pairs)):
                raise ValueError("attacked_task_pairs contains duplicates")
            for user_task_id, injection_task_id in self.attacked_task_pairs:
                if user_task_id not in self.user_tasks:
                    raise ValueError(
                        f"attacked pair references unselected user task {user_task_id!r}"
                    )
                if injection_task_id not in self.injection_tasks:
                    raise ValueError(
                        f"attacked pair references unselected injection task "
                        f"{injection_task_id!r}"
                    )
            if self.run_mode == "benign_only" and self.attacked_task_pairs:
                raise ValueError("benign_only cannot include attacked_task_pairs")
            if self.run_mode != "benign_only" and not self.attacked_task_pairs:
                raise ValueError(
                    f"{self.run_mode} requires at least one explicit attacked_task_pair"
                )
        if self.run_mode != "benign_only" and not self.injection_tasks:
            raise ValueError(f"{self.run_mode} requires explicit injection_tasks")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PilotConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

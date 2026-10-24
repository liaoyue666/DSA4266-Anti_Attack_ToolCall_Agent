from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentdojo.agent_pipeline import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
    chat_completion_request,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import ChatMessage
from openai import OpenAI

from .identifiers import episode_identifier


class DeepSeekLLM(BasePipelineElement):
    """AgentDojo LLM element for DeepSeek's OpenAI-compatible Chat API."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-flash",
        temperature: float | None = 0.0,
        run_id: str | None = None,
        repetition: int | None = None,
        decision: str | None = None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.name = model
        self.run_id = run_id or str(uuid4())
        self.repetition = repetition
        self.decision = decision
        self.telemetry_path = telemetry_path

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None) if usage is not None else None
        return value if isinstance(value, int) else None

    def _record_request(
        self,
        context: dict,
        started_at: float,
        usage: Any = None,
        error: Exception | None = None,
    ) -> None:
        if self.telemetry_path is None:
            return
        raw_context = dict(getattr(Logger.get(), "context", {}))
        duration = time.perf_counter() - started_at
        error_message = str(error) if error is not None else None
        api_key = getattr(self.client, "api_key", None)
        if error_message and isinstance(api_key, str) and api_key:
            error_message = error_message.replace(api_key, "[REDACTED]")
        if usage is None:
            usage_record = None
        elif hasattr(usage, "model_dump"):
            usage_record = usage.model_dump(mode="json")
        elif isinstance(usage, dict):
            usage_record = usage
        else:
            usage_record = {
                "prompt_tokens": self._usage_value(usage, "prompt_tokens"),
                "completion_tokens": self._usage_value(usage, "completion_tokens"),
                "total_tokens": self._usage_value(usage, "total_tokens"),
            }
        record = {
            **context,
            "suite_name": raw_context.get("suite_name"),
            "user_task_id": raw_context.get("user_task_id"),
            "injection_task_id": raw_context.get("injection_task_id"),
            "attack_type": raw_context.get("attack_type"),
            "request_duration_seconds": duration,
            "usage": usage_record,
            "cost": None,
            "error": (
                {
                    "type": type(error).__name__,
                    "message": error_message,
                    "status_code": getattr(error, "status_code", None),
                    "code": getattr(error, "code", None),
                }
                if error is not None
                else None
            ),
        }
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _to_deepseek_message(message: ChatMessage) -> dict:
        converted = dict(_message_to_openai(message, "deepseek-flash"))
        # AgentDojo's OpenAI adapter emits `developer` for its system message.
        # DeepSeek's documented Chat API uses the conventional `system` role.
        if converted.get("role") == "developer":
            converted["role"] = "system"
        return converted

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        api_messages = [self._to_deepseek_message(message) for message in messages]
        api_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        logger_context = dict(getattr(Logger.get(), "context", {}))
        episode_id = episode_identifier(
            self.run_id,
            self.repetition,
            self.decision,
            logger_context.get("suite_name"),
            logger_context.get("user_task_id"),
            logger_context.get("attack_type"),
            logger_context.get("injection_task_id"),
        )
        telemetry_context = {
            "event_id": str(uuid4()),
            "run_id": self.run_id,
            "episode_id": episode_id,
            "repetition": self.repetition,
            "decision_condition": self.decision,
            "model": self.model,
        }
        started_at = time.perf_counter()
        try:
            completion = chat_completion_request(
                self.client,
                self.model,
                api_messages,
                api_tools,
                reasoning_effort=None,
                temperature=self.temperature,
            )
        except Exception as error:
            self._record_request(telemetry_context, started_at, error=error)
            raise
        self._record_request(telemetry_context, started_at, usage=completion.usage)
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from agentdojo.agent_pipeline import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

Decision = Literal["ALLOW", "BLOCK"]


class FixedDecisionGate(BasePipelineElement):
    """Audit consequential tool calls immediately before ``ToolsExecutor``.

    Read-only tools pass through without a decision. Consequential calls are
    logged with the original request, arguments, and latest untrusted tool
    observation. In BLOCK mode, a synthetic tool result is appended so the
    agent can continue without the real tool being executed.
    """

    def __init__(
        self,
        decision: Decision,
        log_path: Path,
        episode_id: str | None = None,
        consequential_tools: set[str] | None = None,
    ) -> None:
        self.decision = decision
        self.log_path = log_path
        self.episode_id = episode_id
        self.consequential_tools = consequential_tools or {"send_email"}

    @staticmethod
    def _latest_untrusted_observation(messages: Sequence[ChatMessage]) -> str | None:
        for message in reversed(messages[:-1]):
            if message["role"] == "tool":
                return get_text_content_as_str(message["content"])
        return None

    @staticmethod
    def collect_observation_provenance(messages: Sequence[ChatMessage]) -> list[dict]:
        """Return all prior tool results with the tool that produced each result."""
        observations = []
        for message in messages:
            if message["role"] != "tool":
                continue
            call = message.get("tool_call")
            observations.append(
                {
                    "tool_name": call.function if call is not None else None,
                    "tool_arguments": dict(call.args) if call is not None else None,
                    "content": get_text_content_as_str(message["content"]),
                    "error": message.get("error"),
                    "untrusted": True,
                }
            )
        return observations

    def _write_record(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls = messages[-1]["tool_calls"] or []
        consequential_calls = [
            call for call in tool_calls if call.function in self.consequential_tools
        ]
        if not consequential_calls:
            return query, runtime, env, messages, extra_args

        observation = self._latest_untrusted_observation(messages)
        observations = self.collect_observation_provenance(messages[:-1])
        for call in consequential_calls:
            self._write_record(
                {
                    "episode_id": self.episode_id,
                    "original_user_request": query,
                    "untrusted_observation": observation,
                    "observation_provenance": observations,
                    "tool_name": call.function,
                    "tool_arguments": dict(call.args),
                    "decision": self.decision,
                    "intercepted_before_execution": True,
                }
            )

        if self.decision == "ALLOW":
            return query, runtime, env, messages, extra_args

        blocked_results = [
            ChatToolResultMessage(
                role="tool",
                content=[
                    text_content_block_from_string(
                        "BLOCKED_BY_AUTHORIZATION_GATE: The proposed action was not executed."
                    )
                ],
                tool_call_id=call.id,
                tool_call=call,
                error="Blocked by fixed authorization policy.",
            )
            for call in consequential_calls
        ]
        return query, runtime, env, [*messages, *blocked_results], extra_args

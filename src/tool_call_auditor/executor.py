from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Callable, Sequence
from pathlib import Path

from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from .gate import Decision, FixedDecisionGate


class AuditedToolsExecutor(FixedDecisionGate):
    """Audit and then individually allow or block each consequential tool call."""

    def __init__(
        self,
        decision: Decision,
        log_path: Path,
        consequential_tools: set[str],
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    ) -> None:
        super().__init__(
            decision=decision,
            log_path=log_path,
            episode_id=None,
            consequential_tools=consequential_tools,
        )
        self.output_formatter = tool_output_formatter

    @staticmethod
    def _episode_context() -> dict:
        logger = Logger.get()
        context = getattr(logger, "context", {})
        return {
            "suite_name": context.get("suite_name"),
            "user_task_id": context.get("user_task_id"),
            "injection_task_id": context.get("injection_task_id"),
            "attack_type": context.get("attack_type"),
            "benchmark_version": context.get("benchmark_version"),
        }

    def _audit(self, query: str, messages: Sequence[ChatMessage], call, executed: bool) -> None:
        observations = self.collect_observation_provenance(messages[:-1])
        context = self._episode_context()
        episode_parts = [
            context.get("suite_name"),
            context.get("user_task_id"),
            context.get("attack_type"),
            context.get("injection_task_id"),
        ]
        record = {
            **context,
            "episode_id": "/".join(str(part or "none") for part in episode_parts),
            "original_user_request": query,
            "untrusted_observation": observations[-1]["content"] if observations else None,
            "observation_provenance": observations,
            "tool_name": call.function,
            "tool_arguments": dict(call.args),
            "decision": self.decision if call.function in self.consequential_tools else "ALLOW_READ_ONLY",
            "consequential": call.function in self.consequential_tools,
            "intercepted_before_execution": True,
            "executed": executed,
        }
        self._write_record(record)

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
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        results = []
        available_tools = {tool.name for tool in runtime.functions.values()}
        for call in tool_calls:
            is_consequential = call.function in self.consequential_tools
            should_block = is_consequential and self.decision == "BLOCK"
            self._audit(query, messages, call, executed=not should_block)

            if should_block:
                results.append(
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
                )
                continue

            if call.function == EMPTY_FUNCTION_NAME or call.function not in available_tools:
                error = (
                    "Empty function name provided. Provide a valid function name."
                    if call.function == EMPTY_FUNCTION_NAME
                    else f"Invalid tool {call.function} provided."
                )
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=call.id,
                        tool_call=call,
                        error=error,
                    )
                )
                continue

            for key, value in call.args.items():
                if isinstance(value, str) and is_string_list(value):
                    call.args[key] = literal_eval(value)
            result, error = runtime.run_function(env, call.function, call.args)
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(self.output_formatter(result))],
                    tool_call_id=call.id,
                    tool_call=call,
                    error=error,
                )
            )

        return query, runtime, env, [*messages, *results], extra_args

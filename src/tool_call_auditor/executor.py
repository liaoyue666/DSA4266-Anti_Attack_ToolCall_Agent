from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

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
from .identifiers import episode_identifier


class AuditedToolsExecutor(FixedDecisionGate):
    """Audit and then individually allow or block each consequential tool call."""

    def __init__(
        self,
        decision: Decision,
        log_path: Path,
        consequential_tools: set[str],
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        run_id: str | None = None,
        repetition: int | None = None,
    ) -> None:
        super().__init__(
            decision=decision,
            log_path=log_path,
            episode_id=None,
            consequential_tools=consequential_tools,
        )
        self.output_formatter = tool_output_formatter
        self.run_id = run_id or str(uuid4())
        self.repetition = repetition

    def _episode_context(self) -> dict:
        logger = Logger.get()
        context = dict(getattr(logger, "context", {}))
        context.update(
            {
                "run_id": self.run_id,
                "repetition": self.repetition,
                "decision_condition": self.decision,
            }
        )
        context["episode_id"] = episode_identifier(
            self.run_id,
            self.repetition,
            self.decision,
            context.get("suite_name"),
            context.get("user_task_id"),
            context.get("attack_type"),
            context.get("injection_task_id"),
        )
        context["task_id"] = context.get("user_task_id")
        context["injection_id"] = context.get("injection_task_id") or (
            context["task_id"]
            if str(context.get("task_id", "")).startswith("injection_task_")
            else None
        )
        context["episode_kind"] = (
            "attack_goal_check"
            if context["injection_id"] == context["task_id"]
            and context["injection_id"] is not None
            else "primary"
        )
        context["scenario_type"] = (
            context["episode_kind"]
            if context["episode_kind"] == "attack_goal_check"
            else "attack"
            if context.get("injection_task_id")
            and context.get("attack_type") not in (None, "none")
            else "normal"
        )
        return context

    def _audit(self, query: str, messages: Sequence[ChatMessage], call, call_id: str) -> dict:
        observations = self.collect_observation_provenance(messages[:-1])
        context = self._episode_context()
        record = {
            **context,
            "event_type": "proposal",
            "call_id": call_id,
            "provider_call_id": call.id,
            "original_user_request": query,
            "untrusted_observation": observations[-1]["content"] if observations else None,
            "observation_provenance": observations,
            "tool_name": call.function,
            "tool_arguments": dict(call.args),
            "gate_decision": (
                self.decision if call.function in self.consequential_tools else "ALLOW_READ_ONLY"
            ),
            "decision": self.decision if call.function in self.consequential_tools else "ALLOW_READ_ONLY",
            "consequential": call.function in self.consequential_tools,
            "intercepted_before_execution": True,
            "label": None,
            "execution_status": "not_attempted",
            "attempted": False,
            "tool_succeeded": None,
            "executed": False,
            "tool_error": None,
            "tool_result": None,
        }
        self._write_record(record)
        return record

    def _write_status(self, record: dict, event_type: str, **status: object) -> None:
        self._write_record({**record, "event_type": event_type, **status})

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
            call_id = f"{self.run_id}:{uuid4()}"
            record = self._audit(query, messages, call, call_id)

            if should_block:
                self._write_status(
                    record,
                    "execution_result",
                    execution_status="blocked",
                    attempted=False,
                    tool_succeeded=False,
                    executed=False,
                    tool_error="Blocked by fixed authorization policy.",
                )
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
                self._write_status(
                    record,
                    "execution_result",
                    execution_status="not_attempted",
                    attempted=False,
                    tool_succeeded=False,
                    executed=False,
                    tool_error=error,
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
            self._write_status(
                record,
                "execution_attempt",
                execution_status="attempted",
                attempted=True,
            )
            try:
                result, error = runtime.run_function(env, call.function, call.args)
            except Exception as exception:
                self._write_status(
                    record,
                    "execution_result",
                    execution_status="error",
                    attempted=True,
                    tool_succeeded=False,
                    executed=False,
                    tool_error=f"{type(exception).__name__}: {exception}",
                )
                raise
            succeeded = error is None
            self._write_status(
                record,
                "execution_result",
                execution_status="succeeded" if succeeded else "error",
                attempted=True,
                tool_succeeded=succeeded,
                executed=succeeded,
                tool_error=error,
                tool_result=self.output_formatter(result) if succeeded else None,
            )
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

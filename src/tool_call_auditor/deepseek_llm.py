from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
    chat_completion_request,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from openai import OpenAI


class DeepSeekLLM(BasePipelineElement):
    """AgentDojo LLM element for DeepSeek's OpenAI-compatible Chat API."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-flash",
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.name = model

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
        completion = chat_completion_request(
            self.client,
            self.model,
            api_messages,
            api_tools,
            reasoning_effort=None,
            temperature=self.temperature,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args

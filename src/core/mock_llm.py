"""A network-free fake chat model for smoke-testing the agent pipeline.

It implements just enough of the LangChain chat-model interface to drive
`langchain.agents.create_agent`: it supports `bind_tools` and emits real
tool calls. For MiniWoB++ it scans the latest observation for the first
element ref and clicks it, then stops after a couple of actions. This lets us
exercise the full agent -> tools -> env -> reward -> accounting -> CSV path
without hitting the (currently down) university LLM endpoint.

Enable it by setting the MOCK_LLM environment variable; `get_llm_client()`
returns this model instead of the real ChatOpenAI client.
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


_REF_RE = re.compile(r"\[(\d+)\]")


class MockToolCallingChat(BaseChatModel):
    """Scripted tool-calling chat model (no network)."""

    max_tool_calls: int = 2
    structured_tool_name: str | None = None

    def bind_tools(self, tools, **kwargs):
        # `with_structured_output` binds the schema as a tool and passes
        # `ls_structured_output_format`. Detect that and remember the schema's
        # tool name so `_generate` can emit a matching structured tool call
        # (used by the B1 planner). Otherwise (executor click/type tools) we
        # ignore the bound list and keep the scripted click behavior.
        if "ls_structured_output_format" in kwargs and tools:
            schema = tools[0]
            name = getattr(schema, "__name__", None) or "output"
            return self.model_copy(update={"structured_tool_name": name})
        return self

    @staticmethod
    def _latest_observation_text(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, (HumanMessage, ToolMessage)):
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _count_prior_tool_calls(messages) -> int:
        return sum(
            1
            for message in messages
            if isinstance(message, AIMessage) and message.tool_calls
        )

    @staticmethod
    def _has_prior_report(messages) -> bool:
        return any(
            isinstance(m, HumanMessage)
            and "Executor report for" in (m.content if isinstance(m.content, str) else "")
            for m in messages
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        # Structured-output path (B1 planner): emit one sub-step, then finish.
        if self.structured_tool_name:
            if self._has_prior_report(messages):
                args = {"status": "done", "substep": ""}
            else:
                args = {
                    "status": "continue",
                    "substep": "Click the first interactable element",
                }
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": self.structured_tool_name, "args": args, "id": "call_planner"}
                ],
                usage_metadata=usage,
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        text = self._latest_observation_text(messages)
        prior_calls = self._count_prior_tool_calls(messages)
        finished = "Episode finished" in text or "already finished" in text
        refs = _REF_RE.findall(text)

        if finished or prior_calls >= self.max_tool_calls or not refs:
            message = AIMessage(content="done", usage_metadata=usage)
        else:
            ref = int(refs[0])
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "click_element",
                        "args": {"ref": ref},
                        "id": f"call_{prior_calls}",
                    }
                ],
                usage_metadata=usage,
            )

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock-tool-calling"

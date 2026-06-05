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

    def bind_tools(self, tools, **kwargs):
        # The script only ever calls click_element, so we ignore the bound
        # tool list and just return self.
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

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self._latest_observation_text(messages)
        prior_calls = self._count_prior_tool_calls(messages)
        finished = "Episode finished" in text or "already finished" in text
        refs = _REF_RE.findall(text)

        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

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

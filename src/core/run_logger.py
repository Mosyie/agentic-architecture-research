"""Per-run LLM I/O logging.

`open_run_log(architecture, difficulty)` creates one log file per run, named
`<timestamp>_<architecture>_<difficulty>.log`, and returns an `LLMIOLogger`
callback handler. Register it via `add_llm_callback` (see
`src/core/llm_client.py`) BEFORE building the agents, and every LLM call --
exactly what is sent and exactly what comes back -- is appended to that file.
"""

import json

from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


LOG_DIR = Path("logs")

TRACE_TEXT_LIMIT = 300


def _format_tool_calls(message) -> list[str]:
    """Render a message's tool calls as `  -> tool_call name({args})` lines.

    Shared by the full file log (`_format_message`) and the compact terminal
    trace (`format_action_trace`) so the two never drift.
    """
    lines = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        args = json.dumps(tool_call.get("args", {}), ensure_ascii=False, default=str)
        lines.append(f"  -> tool_call {tool_call.get('name')}({args})")
    return lines


def _format_message(message) -> str:
    role = getattr(message, "type", message.__class__.__name__)
    content = message.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)

    lines = [f"[{role}]"]
    if content:
        lines.append(content)

    lines.extend(_format_tool_calls(message))

    return "\n".join(lines)


def format_action_trace(messages) -> str:
    """Compact terminal view of an agent run: the tool calls it made plus its
    final reply. Reuses `_format_tool_calls`, but -- unlike the file log -- omits
    the bulky tool-result/observation content so the terminal stays readable.
    """
    lines: list[str] = []
    for message in messages:
        tool_call_lines = _format_tool_calls(message)
        if tool_call_lines:
            lines.extend(tool_call_lines)
        elif isinstance(message, AIMessage):
            text = (message.content or "").strip() if isinstance(message.content, str) else ""
            if text:
                if len(text) > TRACE_TEXT_LIMIT:
                    text = text[: TRACE_TEXT_LIMIT - 1] + "…"
                lines.append(f"  {text}")
    return "\n".join(lines)


class LLMIOLogger(BaseCallbackHandler):
    """Writes every chat-model request and response to a single file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = open(self.path, "a", encoding="utf-8")
        self._call_index = 0

    def _write(self, text: str) -> None:
        self._handle.write(text)
        self._handle.flush()

    def note(self, text: str) -> None:
        """Write a free-form marker (e.g. a task boundary) into the log."""
        self._write(f"\n{'#' * 80}\n# {text}\n{'#' * 80}\n")

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._call_index += 1
        run_id = kwargs.get("run_id")
        stamp = datetime.now().isoformat(timespec="seconds")

        blocks = [
            f"\n{'=' * 80}",
            f"LLM CALL #{self._call_index}  run_id={run_id}  {stamp}",
            f"{'=' * 80}",
        ]

        # `messages` is a list of message lists (one per prompt / generation).
        invocation_params = kwargs.get("invocation_params") or {}
        tools = invocation_params.get("tools")
        tool_choice = invocation_params.get("tool_choice")
        if tools:
            tool_names = [
                t.get("function", {}).get("name", t.get("name", "?"))
                if isinstance(t, dict)
                else str(t)
                for t in tools
            ]
            blocks.append(f"-- bound tools: {tool_names} (tool_choice={tool_choice})")

        for prompt_messages in messages:
            blocks.append(f"-- REQUEST ({len(prompt_messages)} messages) --")
            for message in prompt_messages:
                blocks.append(_format_message(message))

        self._write("\n".join(blocks) + "\n")

    def on_llm_end(self, response, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        blocks = [f"-- RESPONSE  run_id={run_id} --"]

        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is not None:
                    blocks.append(_format_message(message))
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        blocks.append(f"  usage: {json.dumps(usage, default=str)}")
                else:
                    blocks.append(repr(generation.text))

        self._write("\n".join(blocks) + "\n")

    def on_llm_error(self, error, **kwargs) -> None:
        run_id = kwargs.get("run_id")
        self._write(f"-- LLM ERROR  run_id={run_id}: {type(error).__name__}: {error} --\n")

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


def open_run_log(architecture: str, difficulty: str, subdir: str = "miniwob") -> LLMIOLogger:
    """Create a per-run log file and return its LLMIOLogger handler.

    The file is `logs/<subdir>/<timestamp>_<architecture>_<difficulty>.log`.
    """
    log_dir = LOG_DIR / subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{timestamp}_{architecture}_{difficulty}.log"

    logger = LLMIOLogger(path)
    logger._write(
        f"Run log: architecture={architecture} difficulty={difficulty} "
        f"started={datetime.now().isoformat(timespec='seconds')}\n"
    )
    return logger

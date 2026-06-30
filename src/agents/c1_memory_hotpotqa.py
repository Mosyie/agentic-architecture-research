import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_client import get_llm_client
from src.core.metrics    import llm_accounting

from src.agents.b2_actor_critic_hotpotqa import (
    ActorCriticAgent,
    ToolCallEntry,
    _format_trace,
)


REFLECTOR_SYSTEM_PROMPT = (
    "You analyze failures of a question-answering agent and propose ONE generic\n"
    "strategy rule that would have prevented the failure.\n"
    "\n"
    "Constraints:\n"
    "- The rule MUST be entity-free: no names, dates, titles, or specific facts.\n"
    "- The rule MUST be a single sentence.\n"
    "- The rule MUST describe a transferable reasoning or retrieval strategy."
)

CONSOLIDATOR_SYSTEM_PROMPT = (
    "You receive a list of behavioral rules learned from past agent failures.\n"
    "Combine duplicates, drop overly specific or entity-bearing rules, and\n"
    "return the top N most critical generic guidelines for multi-hop QA."
)

DEFAULT_FROZEN_RULES_PATH = "memory/c1_hotpotqa/frozen_rules.json"


class Reflection(BaseModel):
    """One generic, entity-free strategy rule learned from a failed example."""

    rule: str = Field(
        description=(
            "A single sentence describing a transferable reasoning or retrieval "
            "strategy. Must contain no names, dates, titles, or specific facts."
        )
    )


class Consolidation(BaseModel):
    """Compressed top-N list of behavioral rules for multi-hop QA."""

    rules: list[str] = Field(
        description=(
            "Top N most critical generic guidelines, deduplicated, entity-free, "
            "ordered by importance."
        )
    )


def _load_rules(path: Optional[Union[str, Path]]) -> list[str]:
    if path is None:
        return []

    p = Path(path)
    if not p.exists():
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [r for r in data if isinstance(r, str) and r.strip()]

    if isinstance(data, dict):
        rules = data.get("rules", [])
        if isinstance(rules, list):
            return [r for r in rules if isinstance(r, str) and r.strip()]

    return []


def _render_rules_block(rules: list[str]) -> str:
    if not rules:
        return ""

    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
    return (
        "\n\n"
        "Based on past performance analysis, you must strictly adhere to these "
        "proven strategies:\n"
        "<STRATEGY_RULES>\n"
        f"{numbered}\n"
        "</STRATEGY_RULES>"
    )


class MemoryAgent(ActorCriticAgent):
    """Actor/Critic graph with frozen strategy rules injected into both prompts.

    Test-time usage: pass a ``frozen_rules_path`` (default points at the
    standard artefact produced by ``run_train_c1_memory``). If the file is
    missing or contains no rules the constructor raises.

    Training-time usage: pass ``frozen_rules_path=None`` explicitly. The
    agent then runs without a rules block so the trainer can harvest the
    failure traces it needs to produce the rules.
    """

    def __init__(
        self,
        frozen_rules_path: Optional[Union[str, Path]] = DEFAULT_FROZEN_RULES_PATH,
        max_rounds: int = 5,
    ):
        super().__init__(max_rounds=max_rounds)

        self.reflector_llm    = get_llm_client()
        self.consolidator_llm = get_llm_client()

        self.frozen_rules_path = frozen_rules_path

        if frozen_rules_path is None:
            self.rules: list[str] = []
        else:
            p = Path(frozen_rules_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"MemoryAgent: frozen rules file {p!s} does not exist. "
                    f"Run `python -m src.run_train_c1_memory` to produce it, "
                    f"or pass frozen_rules_path=None for training mode."
                )
            rules = _load_rules(frozen_rules_path)
            if not rules:
                raise ValueError(
                    f"MemoryAgent: frozen rules file {p!s} contained no rules. "
                    f"Re-run the trainer or delete the file."
                )
            self.rules = rules

        self._rules_block = _render_rules_block(self.rules)

    @property
    def actor_system_prompt(self) -> str:
        return super().actor_system_prompt + self._rules_block

    @property
    def critic_system_prompt(self) -> str:
        return super().critic_system_prompt + self._rules_block

    def reflect(
        self,
        question: str,
        tool_trace: list[ToolCallEntry],
        prediction: str,
        reference: str,
    ) -> tuple[Optional[str], int, int]:
        """Produce one generic strategy rule from a failed training example.

        Returns ``(rule_or_None, tokens, calls)``. ``rule`` is None when the
        Reflector's structured output could not be obtained.
        """
        trace = _format_trace(tool_trace or [])

        messages = [
            SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Tool Trace:\n{trace}\n\n"
                    f"Agent Answer:\n{prediction}\n\n"
                    f"Ground Truth Answer:\n{reference}\n"
                )
            ),
        ]

        typed = self.reflector_llm.with_structured_output(
            Reflection, method="function_calling", include_raw=True
        )

        try:
            result = typed.invoke(messages)
        except Exception as exc:
            print(f"[REFLECTOR] failed: {type(exc).__name__}: {exc}")
            return None, 0, 0

        ai = result["raw"]
        parsed: Optional[Reflection] = result["parsed"]
        tokens, calls = llm_accounting([ai])

        rule = parsed.rule.strip() if (parsed and parsed.rule) else None
        print(f"[REFLECTOR] {rule!r}")
        return (rule or None), tokens, calls

    def consolidate(
        self,
        rules: list[str],
        top_n: int = 7,
    ) -> tuple[list[str], int, int]:
        """Compress a raw rule list into the top ``top_n`` generic guidelines."""
        if not rules:
            return [], 0, 0

        numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
        messages = [
            SystemMessage(content=CONSOLIDATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Top N: {top_n}\n\n"
                    f"Raw rules ({len(rules)}):\n{numbered}\n"
                )
            ),
        ]

        typed = self.consolidator_llm.with_structured_output(
            Consolidation, method="function_calling", include_raw=True
        )

        try:
            result = typed.invoke(messages)
        except Exception as exc:
            print(f"[CONSOLIDATOR] failed: {type(exc).__name__}: {exc}")
            return [], 0, 0

        ai = result["raw"]
        parsed: Optional[Consolidation] = result["parsed"]
        tokens, calls = llm_accounting([ai])

        if parsed is None or not parsed.rules:
            print(
                "[CONSOLIDATOR] structured output missing; "
                "returning raw rules truncated to top_n"
            )
            return rules[:top_n], tokens, calls

        cleaned = [r.strip() for r in parsed.rules if isinstance(r, str) and r.strip()]
        return cleaned[:top_n], tokens, calls

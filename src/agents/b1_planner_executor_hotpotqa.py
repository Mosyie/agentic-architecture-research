import json

from typing import Literal, NamedTuple, Optional, TypedDict

from pydantic import BaseModel, Field

from langgraph.graph         import END, START, StateGraph
from langchain.agents        import create_agent as _create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.metrics    import llm_accounting
from src.core.llm_client import get_llm_client

EXECUTOR_SYSTEM_PROMPT = (
    "You are an Executor agent. You are given ONE narrow sub-question.\n"
    "Use the available tools to inspect context chunks and find the answer.\n"
    "Answer strictly using information from the tool outputs.\n"
    "Use tools before answering. Read multiple chunks if needed.\n"
    "Return ONLY the short, finalized answer to the sub-question -- no "
    "explanation, no raw chunk text. If the context does not contain the "
    "answer, reply exactly: NOT FOUND."
)


PLANNER_SYSTEM_PROMPT = (
    "You are a Planner agent solving a multi-hop question by delegating.\n"
    "You CANNOT see the source documents and you have NO tools. You work only\n"
    "by issuing one narrow sub-question at a time to an Executor agent, which\n"
    "looks things up and returns a short factual answer.\n"
    "\n"
    "Strategy:\n"
    "- Break the main question into single-hop sub-questions.\n"
    "- Issue ONE sub-question at a time. Wait for the Executor's answer before\n"
    "  deciding the next one (later hops often depend on earlier answers).\n"
    "- Build understanding from the Executor's answers only.\n"
    "- When you have enough information, output the final answer."
)


class PlannerDecision(BaseModel):
    """Planner's next move: either delegate one sub-question, or finish."""

    status: Literal["continue", "done"] = Field(
        description=(
            "'continue' to issue another sub-question, "
            "'done' to return the final answer."
        )
    )
    subquery: str = Field(
        default="",
        description=(
            "When status is 'continue': one narrow sub-question for the "
            "Executor. Empty string when finishing."
        ),
    )
    answer: str = Field(
        default="",
        description=(
            "When status is 'done': the short final answer with no "
            "explanation. Empty string when continuing."
        ),
    )


class SubqueryLogEntry(NamedTuple):
    subquery: str
    answer: str


class PlannerExecutorState(TypedDict, total=False):
    question: str
    subquery_log: list[SubqueryLogEntry]
    next_subquery: str
    final_answer: str
    total_tokens: int
    num_api_calls: int
    num_subqueries: int
    planner_steps: int
    error: str


class PlannerExecutorAgent:
    """Two-node planner/executor graph for multi-hop question answering."""

    def __init__(
        self,
        max_planner_steps: int = 8,
        executor_max_steps: int = 10,
    ):
        self.max_planner_steps = max_planner_steps
        self.executor_max_steps = executor_max_steps
        self.planner_llm  = get_llm_client()
        self.executor_llm = get_llm_client()

    def _planner_node(self, state: PlannerExecutorState) -> dict:
        """Ask the planner for either the next subquery or the final answer.

        On any failure (LLM error, unparseable reply) this returns an `error`
        and no `next_subquery`/`final_answer`. The router keys off those, so
        there is no need to fabricate a "done" decision to stop the graph.
        """
        msgs = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Main question: {state['question']}"),
        ]

        for entry in state.get("subquery_log", []):
            msgs.append(
                AIMessage(
                    content=json.dumps(
                        {"status": "continue", "subquery": entry.subquery}
                    )
                )
            )
            msgs.append(
                HumanMessage(
                    content=(
                        f"Executor answer to '{entry.subquery}': {entry.answer}\n"
                        "Decide the next sub-question or finish."
                    )
                )
            )

        typed = self.planner_llm.with_structured_output(
            PlannerDecision, method="json_schema", include_raw=True
        )

        try:
            result = typed.invoke(msgs)
        except Exception as exc:
            return {
                "error": f"Planner call failed: {type(exc).__name__}: {exc}",
            }

        ai = result["raw"]
        decision: Optional[PlannerDecision] = result["parsed"]
        print(f"\nPLANNER: {decision!r}")

        planner_tokens, planner_calls = llm_accounting([ai])

        updates = {
            "total_tokens": state.get("total_tokens", 0) + planner_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + planner_calls,
            "planner_steps": state.get("planner_steps", 0) + 1,
        }

        if decision is None:
            updates["error"] = "Planner returned invalid structured output."
            return updates

        if decision.status == "done":
            answer = (decision.answer or "").strip()
            if not answer:
                updates["error"] = "Planner finished with an empty answer."
            else:
                updates["final_answer"] = answer
        else:
            subquery = (decision.subquery or "").strip()
            updates["next_subquery"] = subquery

        return updates

    def _executor_node(self, state: PlannerExecutorState, executor_agent) -> dict:
        """Run the tool-using executor on the current subquery."""
        subquery = state.get("next_subquery", "").strip()

        if not subquery:
            return {
                "subquery_log": state.get("subquery_log", [])
                + [SubqueryLogEntry("(empty sub-query)", "NOT FOUND")],
                "num_subqueries": state.get("num_subqueries", 0) + 1,
                "error": "Planner produced an empty subquery.",
            }

        print(f"--> DELEGATING SUB-QUERY: {subquery}")

        ans = "NOT FOUND"
        exec_tokens = 0
        exec_calls = 0

        try:
            result = executor_agent.invoke(
                {"messages": [HumanMessage(content=subquery)]},
                config={"recursion_limit": self.executor_max_steps * 2 + 1},
            )

            messages = result.get("messages", [])
            exec_tokens, exec_calls = llm_accounting(messages)

            for message in reversed(messages):
                if isinstance(message, AIMessage) and not getattr(
                    message, "tool_calls", None
                ):
                    ans = (message.content or "").strip()
                    break

        except Exception as exc:
            return {
                "subquery_log": state.get("subquery_log", [])
                + [SubqueryLogEntry(subquery, "NOT FOUND")],
                "total_tokens": state.get("total_tokens", 0) + exec_tokens,
                "num_api_calls": state.get("num_api_calls", 0) + exec_calls,
                "num_subqueries": state.get("num_subqueries", 0) + 1,
                "error": f"Executor failed: {type(exc).__name__}: {exc}",
            }

        print(f"<-- EXECUTOR ANSWER: {ans}")

        return {
            "subquery_log": state.get("subquery_log", [])
            + [SubqueryLogEntry(subquery, ans)],
            "total_tokens": state.get("total_tokens", 0) + exec_tokens,
            "num_api_calls": state.get("num_api_calls", 0) + exec_calls,
            "num_subqueries": state.get("num_subqueries", 0) + 1,
        }

    def _route_after_planner(self, state: PlannerExecutorState) -> str:
        """Route to executor unless we errored, finished, or hit the step guard."""
        if state.get("error"):
            return "end"

        if state.get("final_answer") is not None:
            return "end"

        if state.get("planner_steps", 0) >= self.max_planner_steps:
            return "end"

        return "executor"

    def _build_graph(self, executor_agent):
        graph = StateGraph(PlannerExecutorState)

        graph.add_node("planner", self._planner_node)

        def executor_node(state: PlannerExecutorState) -> dict:
            return self._executor_node(state, executor_agent)

        graph.add_node("executor", executor_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "executor": "executor",
                "end": END,
            },
        )
        graph.add_edge("executor", "planner")

        return graph.compile()

    def invoke(self, question: str, tools) -> dict:
        """Run the planner/executor graph for one question."""
        executor_agent = _create_agent(
            self.executor_llm,
            tools,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
        )

        graph = self._build_graph(executor_agent)

        initial_state: PlannerExecutorState = {
            "question": question,
            "subquery_log": [],
            "total_tokens": 0,
            "num_api_calls": 0,
            "num_subqueries": 0,
            "planner_steps": 0,
        }

        # Each planner step is up to two graph nodes (planner -> executor), so
        # the graph-level recursion budget is ~2x the planner-step budget plus
        # slack for the final planner pass and routing.
        config = {"recursion_limit": self.max_planner_steps * 2 + 5}

        try:
            final_state = graph.invoke(initial_state, config=config)
        except Exception as exc:
            return {
                "answer": "",
                "total_tokens": 0,
                "num_api_calls": 0,
                "num_subqueries": 0,
                "error": f"Graph failed: {type(exc).__name__}: {exc}",
            }

        answer = final_state.get("final_answer", "")
        error = final_state.get("error", "")

        if not answer and not error:
            if final_state.get("planner_steps", 0) >= self.max_planner_steps:
                error = "Maximum planner steps reached before final answer."
            else:
                error = "No final answer produced."

        output = {
            "answer": answer,
            "total_tokens": final_state.get("total_tokens", 0),
            "num_api_calls": final_state.get("num_api_calls", 0),
            "num_subqueries": final_state.get("num_subqueries", 0),
        }

        if error:
            output["error"] = error

        return output

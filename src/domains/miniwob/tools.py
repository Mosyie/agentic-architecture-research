from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from src.domains.miniwob.env import (
    format_observation,
    execute_click,
    execute_type,
)


class ClickElementArgs(BaseModel):
    ref: int = Field(description="The ref id of the DOM element to click.")


class TypeTextArgs(BaseModel):
    text: str = Field(description="The text to type into the input element.")
    ref: int = Field(description="The ref id of the input element to focus and type into.")


def make_miniwob_tools(env, episode_state: dict) -> list[StructuredTool]:
    """Build click/type tools that step `env` and record progress in `episode_state`.

    `episode_state` is the runner-owned mutable dict:
        {"last_reward": 0.0, "terminated": False, "truncated": False, "num_actions": 0}
    """

    def _apply(step_result) -> str:
        obs, reward, terminated, truncated, _info = step_result

        episode_state["num_actions"] += 1
        episode_state["last_reward"] = float(reward)
        episode_state["terminated"] = bool(terminated)
        episode_state["truncated"] = bool(truncated)

        if terminated or truncated:
            return (
                f"Episode finished (reward={float(reward):.3f}). "
                "No further actions possible."
            )

        return format_observation(obs)

    def click_element(ref: int) -> str:
        if episode_state["terminated"] or episode_state["truncated"]:
            return "Episode already finished."

        try:
            return _apply(execute_click(env, ref))
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def type_text(text: str, ref: int) -> str:
        if episode_state["terminated"] or episode_state["truncated"]:
            return "Episode already finished."

        try:
            return _apply(execute_type(env, text, ref))
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    return [
        StructuredTool.from_function(
            func=click_element,
            name="click_element",
            description="Click the DOM element with the given ref.",
            args_schema=ClickElementArgs,
        ),
        StructuredTool.from_function(
            func=type_text,
            name="type_text",
            description="Focus the input element with the given ref and type text into it.",
            args_schema=TypeTextArgs,
        ),
    ]

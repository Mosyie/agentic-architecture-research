import functools

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from src.domains.miniwob.env import (
    format_observation,
    execute_click,
    execute_type,
    execute_press_key,
    execute_drag,
)


class ClickElementArgs(BaseModel):
    ref: int = Field(description="The ref id of the DOM element to click.")


class TypeTextArgs(BaseModel):
    text: str = Field(description="The text to type into the input element.")
    ref: int = Field(description="The ref id of the input element to focus and type into.")


class DragArgs(BaseModel):
    sx: int = Field(description="X pixel to press the mouse down at (pickup).")
    sy: int = Field(description="Y pixel to press the mouse down at (pickup).")
    tx: int = Field(description="X pixel to release the mouse at (drop).")
    ty: int = Field(description="Y pixel to release the mouse at (drop).")


class PressKeyArgs(BaseModel):
    key: str = Field(
        description=(
            "Exact key token to press on the focused element. Special keys are "
            "angle-bracketed: <Enter>, <Backspace>, <Tab>, <Space>, <Delete>, "
            "<ArrowUp>, <ArrowDown>, <ArrowLeft>, <ArrowRight>. Ctrl combos are not "
            "bracketed: C-a, C-c, C-x, C-v. Use type_text for characters."
        )
    )


def make_miniwob_tools(env, episode_state: dict) -> list[StructuredTool]:
    """Build the LLM tools for one episode, bound to its `env` and `episode_state`."""

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

    # max_concurrency=1 (agent config) serializes batched tool calls, so the
    # single browser is never driven in parallel.
    def _tool(action):
        """Guard if the episode ended, apply the step result, surface errors as text."""
        @functools.wraps(action)
        def wrapped(*args, **kwargs):
            if episode_state["terminated"] or episode_state["truncated"]:
                return "Episode already finished."
            try:
                return _apply(action(*args, **kwargs))
            except Exception as exc:
                return f"Error: {type(exc).__name__}: {exc}"
        return wrapped

    @_tool
    def click_element(ref: int):
        return execute_click(env, ref)

    @_tool
    def type_text(text: str, ref: int):
        return execute_type(env, text, ref)

    @_tool
    def press_key(key: str):
        return execute_press_key(env, key)

    @_tool
    def drag(sx: int, sy: int, tx: int, ty: int):
        return execute_drag(env, sx, sy, tx, ty)

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
        StructuredTool.from_function(
            func=press_key,
            name="press_key",
            description="Press a key token on the currently focused element.",
            args_schema=PressKeyArgs,
        ),
        StructuredTool.from_function(
            func=drag,
            name="drag",
            description=(
                "Drag from one page point to another: press at (sx, sy) and release "
                "at (tx, ty), in page pixels matching the shown (left,top widthxheight). "
                "To grab or drop on an element, aim at its center: "
                "(left + width/2, top + height/2)."
            ),
            args_schema=DragArgs,
        ),
    ]

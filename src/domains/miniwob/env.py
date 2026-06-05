import gymnasium
import miniwob  # noqa: F401  (importing registers the miniwob/* environments)

from miniwob.action import ActionTypes


MAX_ELEMENTS = 60
TEXT_TRUNCATE = 60


def task_to_env_id(task: str) -> str:
    """Map a MiniWoB++ task name to its Gymnasium env id."""
    return f"miniwob/{task}-v1"


def make_env(task: str, headless: bool = True, wait_ms: int | None = None) -> gymnasium.Env:
    """Create a MiniWoB++ environment for one task.

    MiniWoB derives headless mode from render_mode: None => headless Chrome,
    "human" => a visible Chrome window. By default we pause the browser after
    each action only when it is visible (so a "--no-headless" run is watchable),
    and not at all when headless (so benchmark runs stay fast). The default
    action-space config already supports CLICK_ELEMENT and the text-typing
    actions, so no custom config is needed.
    """
    render_mode = None if headless else "human"
    if wait_ms is None:
        wait_ms = 0 if headless else 500
    return gymnasium.make(task_to_env_id(task), render_mode=render_mode, wait_ms=wait_ms)


def reset_env(env: gymnasium.Env, seed: int | None = None):
    """Reset the env, returning (obs, info)."""
    return env.reset(seed=seed)


def _field(element, name):
    """Read a field from a DOM element that may be a dict or a numpy record."""
    try:
        if isinstance(element, dict):
            return element.get(name)
        return element[name]
    except (KeyError, IndexError, ValueError):
        return None


def _to_float(value):
    """Coerce a scalar or numpy array (MiniWoB stores sizes as length-1 arrays)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        item = getattr(value, "item", None)
        if item is not None:
            try:
                return float(item())
            except (TypeError, ValueError):
                return None
        try:
            return float(value[0])
        except (TypeError, ValueError, IndexError, KeyError):
            return None


def _truncate(value, limit: int = TEXT_TRUNCATE) -> str:
    text = str(value).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _interactable_elements(obs) -> list[dict]:
    """Keep visible, actionable DOM elements with a positive ref, capped for prompt size."""
    elements = obs.get("dom_elements", []) if isinstance(obs, dict) else []

    kept = []
    for element in elements:
        ref = _to_float(_field(element, "ref"))
        width = _to_float(_field(element, "width"))
        height = _to_float(_field(element, "height"))

        if ref is None or width is None or height is None:
            continue

        ref = int(ref)
        if ref <= 0 or width <= 0 or height <= 0:
            continue

        kept.append(
            {
                "ref": ref,
                "tag": _truncate(_field(element, "tag") or ""),
                "text": _truncate(_field(element, "text") or ""),
                "value": _truncate(_field(element, "value") or ""),
            }
        )

        if len(kept) >= MAX_ELEMENTS:
            break

    return kept


def format_observation(obs) -> str:
    """Render the utterance + a numbered list of interactable DOM elements."""
    utterance = obs.get("utterance", "") if isinstance(obs, dict) else ""

    lines = [f"Instruction: {utterance}", ""]
    lines.append('Interactable elements (ref: tag "text" value=<value>):')

    elements = _interactable_elements(obs)
    if not elements:
        lines.append("(no interactable elements found)")
    else:
        for element in elements:
            lines.append(
                f'[{element["ref"]}] {element["tag"]} '
                f'"{element["text"]}" value={element["value"]}'
            )

    return "\n".join(lines)


def execute_click(env: gymnasium.Env, ref: int):
    """Click the element with the given ref. Returns the step tuple."""
    action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=int(ref))
    return env.step(action)


def execute_type(env: gymnasium.Env, text: str, ref: int | None = None):
    """Type text, focusing the element with the given ref first when provided."""
    if ref is not None:
        action = env.unwrapped.create_action(
            ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=int(ref), text=str(text)
        )
    else:
        action = env.unwrapped.create_action(ActionTypes.TYPE_TEXT, text=str(text))
    return env.step(action)

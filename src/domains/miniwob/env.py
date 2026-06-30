import numpy as np

import gymnasium
import miniwob  # noqa: F401  (importing registers the miniwob/* environments)

from miniwob.action import ActionTypes
from miniwob.reward import get_raw_reward


# Tags the agent can act on. Other elements are kept only if they carry text.
INTERACTIVE_TAGS = {
    "button", "a", "select", "option", "textarea", "label", "checkbox", "radio",
}


def _is_interactive(tag: str) -> bool:
    tag = (tag or "").lower()
    return tag.startswith("input") or tag in INTERACTIVE_TAGS


# An LLM in the loop easily exceeds MiniWoB's default 10s episode timer, so we
# raise it to 10 minutes; with get_raw_reward the score is latency-independent.
DEFAULT_EPISODE_MAX_TIME_MS = 600_000


def _set_episode_time_limit(env: gymnasium.Env, ms: int) -> None:
    """Raise the in-browser episode timer before the first episode starts."""
    inst = env.unwrapped.instance
    inst.call(inst.driver.execute_script, f"core.EPISODE_MAX_TIME = {int(ms)};")
    inst.wait()


def make_env(task: str, headless: bool = True) -> gymnasium.Env:
    """Create a MiniWoB++ environment for one task (headless or a visible window)."""
    env = gymnasium.make(
        f"miniwob/{task}-v1",
        render_mode=None if headless else "human",
        wait_ms=0 if headless else 500,  # pause between actions only when visible
        reward_processor=get_raw_reward,  # score independent of LLM latency
    )
    _set_episode_time_limit(env, DEFAULT_EPISODE_MAX_TIME_MS)
    return env


def _to_float(value):
    """Coerce MiniWoB geometry to a float. left/top/width/height come back as
    length-1 np.float32 arrays; ref/parent are already plain ints."""
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _child_text_by_parent(elements) -> dict:
    """Map each parent ref -> the text in its child 't' nodes (MiniWoB keeps many
    labels in a separate 't' node pointing back at the element)."""
    by_parent: dict = {}
    for element in elements:
        if str(element["tag"] or "").strip().lower() != "t":
            continue
        text = str(element["text"] or "").strip()
        if not text:
            continue
        parent = int(element["parent"])
        by_parent.setdefault(parent, []).append(text)
    return {ref: " ".join(parts) for ref, parts in by_parent.items()}


def _parent_refs(elements) -> set:
    """Refs that are a parent of some node, i.e. non-leaf (container) elements."""
    parents = set()
    for element in elements:
        parents.add(int(element["parent"]))
    return parents


def _interactable_elements(obs) -> list[dict]:
    """Keep the visible, actionable elements and drop layout/container junk: an
    element survives if it has a positive ref and non-zero size and is interactive,
    carries text, or is a leaf control with an id/class (e.g. a CSS-glyph icon)."""
    elements = obs["dom_elements"]
    child_text = _child_text_by_parent(elements)
    parent_refs = _parent_refs(elements)

    kept = []
    for element in elements:
        ref = int(element["ref"])
        left = _to_float(element["left"])
        top = _to_float(element["top"])
        width = _to_float(element["width"])
        height = _to_float(element["height"])

        if ref <= 0 or width <= 0 or height <= 0:
            continue

        tag = str(element["tag"] or "").strip()
        if tag.lower() == "t":
            continue  # text node; its text is folded into its parent

        own_text = str(element["text"] or "").strip()
        extra = child_text.get(ref, "")
        text = f"{own_text} {extra}".strip() if extra else own_text

        elem_id = element["id"]
        classes = element["classes"]

        # A leaf (no child nodes) with an id or class but no text is an icon /
        # control rendered via CSS; keep it so the agent can click it.
        is_control = ref not in parent_refs and (elem_id or classes)

        if not (_is_interactive(tag) or text or is_control):
            continue

        kept.append(
            {
                "ref": ref,
                "tag": tag,
                "elem_id": elem_id,
                "classes": classes,
                "text": text,
                "value": element["value"],
                "left": int(round(left)),
                "top": int(round(top)),
                "width": int(round(width)),
                "height": int(round(height)),
            }
        )

    return kept


def _svg_fills(env: gymnasium.Env) -> dict[int, str]:
    """Map ref -> explicit SVG `fill` attribute for the live page.

    MiniWoB's DOM extraction only captures CSS background/foreground colors,
    which are transparent/black for SVG shapes -- the color that tasks like
    drag-shapes-2 grade on lives in the `fill` attribute, so we read it from
    the browser directly. Elements without an explicit fill (and containers
    with fill='none') are omitted.
    """
    fills = env.unwrapped.instance.driver.execute_script(
        "var out = {};"
        "document.querySelectorAll('[data-wob_ref]').forEach(function(e){"
        "  var f = e.getAttribute('fill');"
        "  if (f && f !== 'none') out[e.getAttribute('data-wob_ref')] = f;"
        "});"
        "return out;"
    )
    return {int(ref): str(fill) for ref, fill in (fills or {}).items()}


def _format_element(element: dict, fills: dict[int, str]) -> str:
    parts = [f'[{element["ref"]}]', element["tag"]]
    if element["elem_id"]:
        parts.append(f'#{element["elem_id"]}')
    if element["text"]:
        parts.append(f'"{element["text"]}"')
    elif element["classes"]:
        parts.append(f'class="{element["classes"]}"')  # icons/controls with no text
    if element["value"]:
        parts.append(f'value="{element["value"]}"')
    fill = fills.get(element["ref"])
    if fill:
        parts.append(f'fill="{fill}"')
    parts.append(f'({element["left"]},{element["top"]} {element["width"]}x{element["height"]})')
    return " ".join(parts)


def format_observation(obs, env: gymnasium.Env | None = None) -> str:
    """Render the utterance + a list of interactable DOM elements.

    Pass `env` to annotate SVG shapes with their fill color (see _svg_fills);
    without it the rendering is unchanged.
    """
    utterance = obs["utterance"]

    lines = [
        f"Instruction: {utterance}",
        "",
        'Interactable elements ([ref] tag #id "text" value="..." '
        "(left,top widthxheight)):",
    ]

    fills = _svg_fills(env) if env is not None else {}

    elements = _interactable_elements(obs)
    if not elements:
        lines.append("(no interactable elements found)")
    else:
        lines.extend(_format_element(element, fills) for element in elements)

    return "\n".join(lines)


def execute_click(env: gymnasium.Env, ref: int):
    """Click the element with the given ref. Returns the step tuple."""
    action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=int(ref))
    return env.step(action)


def _clear_element(env: gymnasium.Env, ref: int) -> None:
    """Empty the field by ref so the next type overwrites instead of appending
    (MiniWoB's type action appends to the current contents)."""
    inst = env.unwrapped.instance
    inst.driver.execute_script(
        "var e = document.querySelector('[data-wob_ref=\"' + arguments[0] + '\"]');"
        "if (e && 'value' in e) { e.value = ''; }",
        int(ref),
    )


def execute_type(env: gymnasium.Env, text: str, ref: int):
    """Clear the field with the given ref, then focus it and type the text."""
    _clear_element(env, ref)
    action = env.unwrapped.create_action(
        ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=int(ref), text=str(text)
    )
    return env.step(action)


def execute_press_key(env: gymnasium.Env, key: str):
    """Press a key on the focused element. `key` must be an exact token from the
    env's allowed_keys (e.g. '<Enter>', '<Tab>', 'C-c'); PRESS_KEY takes its index.
    Raises ValueError listing the valid keys on a miss, which the tool layer relays
    back to the agent so it can retry with a correct token."""
    allowed = list(env.unwrapped.action_space_config.allowed_keys)
    if key not in allowed:
        raise ValueError(f"Unsupported key {key!r}. Allowed keys: {', '.join(allowed)}")
    action = env.unwrapped.create_action(ActionTypes.PRESS_KEY, key=allowed.index(key))
    return env.step(action)


def _coords_step(env: gymnasium.Env, action_type, x: float, y: float):
    action = env.unwrapped.create_action(
        action_type, coords=np.array([x, y], dtype=np.float32)
    )
    return env.step(action)


def execute_drag(env: gymnasium.Env, sx, sy, tx, ty, steps: int = 6):
    """Drag from (sx, sy) to (tx, ty): press, glide to the target over `steps`
    interpolated moves, release; returns the final step tuple. The moves matter --
    jQuery draggable only engages while the held pointer is moving, so a single jump
    would not register as a drag."""
    _coords_step(env, ActionTypes.MOUSEDOWN_COORDS, sx, sy)
    for i in range(1, steps + 1):
        _coords_step(
            env,
            ActionTypes.MOVE_COORDS,
            sx + (tx - sx) * i / steps,
            sy + (ty - sy) * i / steps,
        )
    return _coords_step(env, ActionTypes.MOUSEUP_COORDS, tx, ty)

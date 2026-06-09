

hard = [
    "drag-sort-numbers",
    "tic-tac-toe",
    "use-spinner",
    "book-flight",
    "click-checkboxes-soft",
    "click-menu-2",
    "copy-paste-2",
    "daily-calendar",
    "terminal",
    "social-media",
    "guess-number",
    "email-inbox",
    "order-food"
    ]

medium = [
    "drag-shapes",
    "drag-shapes-2",
    "copy-paste",
    "click-checkboxes-large",
    "click-collapsible-2",
    "click-widget",
    "email-inbox-forward",
    "use-autocomplete",
    "search-engine",
    "navigate-tree"
    ]

easy = [
    "drag-box",
    "drag-circle",
    "drag-single-shape",
    "grid-coordinate",
    "login-user",
    "click-button",
    "click-menu",
    "email-inbox-delete",
    "text-transform",
    "simple-arithmetic",
    "simple-algebra",
    "read-table-2",
    "focus-text-2"
    ]

"""
Difficulty is rated on the task itself (architecture-agnostic), against the
text/DOM observation we feed the agent -- NOT pixel/vision difficulty. Five axes:

  1. horizon      -- number of dependent actions before reward
  2. statefulness -- must the agent react to mid-task UI changes (dropdown opens,
                     autocomplete populates, tree expands) or is it one-shot?
  3. reasoning    -- arithmetic / algebra / sorting / semantic-matching / strategy
                     vs. a direct lookup
  4. planning     -- independent sub-goals with ordering constraints
                     (e.g. book-flight: search -> filter -> pick -> confirm)
  5. grounding    -- is the needed info directly legible, or must the agent
                     discover a non-obvious interaction (spinner needs arrows)?

  easy   = low on all five (horizon 1-3, no real reasoning, one-shot)
  medium = bumps exactly ONE axis (longer horizon, OR statefulness, OR reasoning)
  hard   = bumps TWO+ axes -- esp. planning + reasoning, or a feedback loop where
           errors compound

Notes:
- removed "choose-date" (for now?) -> giant token usage, run only once if possible, task is working fine
- drag is coordinate-based (the agent supplies pickup/drop pixels): drag-sort-numbers's
  sortable list reflows after each move, so coords from an earlier observation go stale
  -- may cause issues unless the agent re-reads positions between drags
"""

_LEVELS = {"easy": easy, "medium": medium, "hard": hard}


def get_tasks(level: str) -> list[str]:
    """Return the MiniWoB++ task names for a difficulty level."""
    key = level.lower()
    if key not in _LEVELS:
        raise ValueError(f"Unknown level '{level}'. Use easy | medium | hard.")
    return list(_LEVELS[key])


if __name__ == "__main__":
    print(f"easy: {len(easy)}")
    print(f"medium: {len(medium)}")
    print(f"hard: {len(hard)}")
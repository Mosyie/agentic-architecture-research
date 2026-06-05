
hard = ["bisect-angle", "book-flight", "click-checkboxes-soft", "click-menu-2", "copy-paste-2", "daily-calendar", "drag-shapes-2", "use-colorwheel-2", "terminal", "social-media", "number-checkboxes", "guess-number", "email-inbox", "order-food", "stock-market"]

medium = ["click-checkboxes-large", "click-collapsible-2", "click-widget", "copy-paste", "email-inbox-forward", "visual-addition", "use-slider-2", "use-colorwheel", "use-autocomplete", "tic-tac-toe", "search-engine", "right-angle", "navigate-tree", "highlight-text-2", "grid-coordinate"]

easy = ["choose-date", "choose-list", "click-button", "click-menu", "email-inbox-delete", "use-spinner", "use-slider", "text-transform", "text-editor", "simple-arithmetic", "simple-algebra", "resize-textarea", "read-table-2", "login-user", "focus-text-2"]

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
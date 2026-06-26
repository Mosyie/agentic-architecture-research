import json
from pathlib import Path

import pandas as pd

from src.domains.miniwob.loader import get_tasks
from src.domains.miniwob.env    import make_env, format_observation
from src.domains.miniwob.tools  import make_miniwob_tools
from src.agents.c1_memory_miniwob import MemoryAgent


# --- CONFIGURATION CONSTANTS ---
LEVELS                      = ("easy", "medium", "hard")
TASKS_PER_LEVEL             = 15
REFLECTION_REWARD_THRESHOLD = 1.0   # reflect on every non-perfect episode (reward < 1.0)
TOP_N                       = 7
TRAIN_SEED                  = 41    # single seed, disjoint from the test seeds {42,43,44}
MEMORY_DIR                  = "memory/c1_miniwob"


def build_training_tasks(tasks_per_level: int) -> tuple[list[tuple[str, str]], dict]:
    """Return [(level, task), ...] using the first ``tasks_per_level`` per level.

    Also returns the per-level task counts.
    """
    pairs: list[tuple[str, str]] = []
    per_level_counts: dict[str, int] = {}

    for level in LEVELS:
        tasks = get_tasks(level)[:tasks_per_level]
        per_level_counts[level] = len(tasks)
        pairs.extend((level, task) for task in tasks)

    if not pairs:
        raise RuntimeError(
            "No training tasks could be collected -- check get_tasks() output."
        )

    return pairs, per_level_counts


def save_raw_rules(path: Path, rules: list[str]) -> None:
    path.write_text(
        json.dumps({"rules": rules}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    memory_dir = Path(MEMORY_DIR)
    memory_dir.mkdir(parents=True, exist_ok=True)

    raw_rules_path = memory_dir / "raw_rules.json"
    frozen_path    = memory_dir / "frozen_rules.json"
    log_path       = memory_dir / "training_log.csv"

    print(
        f"[train] Building training tasks "
        f"(tasks_per_level={TASKS_PER_LEVEL}, train_seed={TRAIN_SEED})..."
    )

    train_pairs, per_level_counts = build_training_tasks(TASKS_PER_LEVEL)

    print(
        f"[train] Collected {len(train_pairs)} tasks "
        f"(per level: {per_level_counts})."
    )

    agent = MemoryAgent(frozen_rules_path=None)

    raw_rules: list[str] = []
    log_rows: list[dict] = []

    total_tokens = 0
    total_calls  = 0

    for index, (level, task) in enumerate(train_pairs):
        print(
            f"\n--- Train episode {index + 1}/{len(train_pairs)} "
            f"(level={level}, task={task}, seed={TRAIN_SEED}) ---"
        )

        env = None
        rule_added = False
        try:
            env = make_env(task, headless=True)
            obs, _info = env.reset(seed=TRAIN_SEED)

            instruction = format_observation(obs, env)

            episode_state = {
                "last_reward": 0.0,
                "terminated": False,
                "truncated": False,
                "num_actions": 0,
            }
            tools = make_miniwob_tools(env, episode_state)

            result = agent.invoke(
                instruction=instruction,
                tools=tools,
            )

            reward        = episode_state["last_reward"]
            num_actions   = episode_state["num_actions"]
            tokens_used   = result.get("total_tokens", 0)
            num_api_calls = result.get("num_api_calls", 0)
            error         = result.get("error", "")

            total_tokens += tokens_used
            total_calls  += num_api_calls

            if reward < REFLECTION_REWARD_THRESHOLD:
                rule, r_tokens, r_calls = agent.reflect(
                    instruction=instruction,
                    action_trace=result.get("action_trace", []),
                    actor_report=result.get("actor_report", ""),
                    current_observation=result.get("current_observation", ""),
                    reward=reward,
                )
                total_tokens += r_tokens
                total_calls  += r_calls

                if rule:
                    raw_rules.append(rule)
                    save_raw_rules(raw_rules_path, raw_rules)
                    rule_added = True
                    print(f"[+rule] {rule}")
                else:
                    print("[reflector] failed to produce a usable rule.")
        except Exception as exc:
            reward        = 0.0
            num_actions   = 0
            tokens_used   = 0
            num_api_calls = 0
            error         = f"{type(exc).__name__}: {exc}"
            print(f"Error: {error}")
        finally:
            if env is not None:
                env.close()

        log_rows.append(
            {
                "level":         level,
                "task":          task,
                "seed":          TRAIN_SEED,
                "reward":        reward,
                "num_actions":   num_actions,
                "tokens_used":   tokens_used,
                "num_api_calls": num_api_calls,
                "error":         error,
                "rule_added":    rule_added,
            }
        )
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        print(f"Reward: {reward:.3f} | Actions: {num_actions}")
        print(f"Tokens: {tokens_used} | API Calls: {num_api_calls}")

    raw_rule_count = len(raw_rules)
    print(
        f"\n[train] Training pass complete. "
        f"Raw rules harvested: {raw_rule_count}. "
        f"Running consolidator (top_n={TOP_N})..."
    )

    consolidated, c_tokens, c_calls = agent.consolidate(
        rules=raw_rules,
        top_n=TOP_N,
    )
    total_tokens += c_tokens
    total_calls  += c_calls

    frozen_payload = {
        "rules":                consolidated,
        "training_episodes":    len(train_pairs),
        "raw_rule_count":       raw_rule_count,
        "consolidator_top_n":   TOP_N,
        "per_level_counts":     per_level_counts,
        "tasks_per_level":      TASKS_PER_LEVEL,
        "train_seed":           TRAIN_SEED,
        "reflection_threshold": REFLECTION_REWARD_THRESHOLD,
        "total_tokens":         total_tokens,
        "total_api_calls":      total_calls,
    }
    frozen_path.write_text(
        json.dumps(frozen_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[train] Wrote {len(consolidated)} frozen rules to: {frozen_path}")
    print(f"[train] Raw rules:    {raw_rules_path}")
    print(f"[train] Training log: {log_path}")
    print(f"[train] Total tokens: {total_tokens} | API calls: {total_calls}")


if __name__ == "__main__":
    main()

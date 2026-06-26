import json
from pathlib import Path

import pandas as pd

from src.core.metrics            import exact_match, f1_score_text
from src.domains.hotpotqa.tools  import make_hotpotqa_tools
from src.domains.hotpotqa.loader import load_hotpotqa_sample
from src.agents.c1_memory_hotpotqa import MemoryAgent


# --- CONFIGURATION CONSTANTS ---
LEVELS = ("easy", "medium", "hard")
REFLECTION_F1_THRESHOLD = 1.0  # reflect on every non-perfect answer (f1 < 1.0)
TEST_SET_SIZE = 500
TOP_N = 7
SEED = 42
NUM_TRAIN_SAMPLES = 100
MEMORY_DIR = "memory/c1_hotpotqa"


def split_counts(total: int, n_buckets: int) -> list[int]:
    base = total // n_buckets
    rem = total % n_buckets
    return [base + (1 if i < rem else 0) for i in range(n_buckets)]


def build_training_set(
    test_set_size: int,
    num_train_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    per_level = split_counts(num_train_samples, len(LEVELS))
    pieces = []
    per_level_counts: dict[str, int] = {}

    for level, wanted in zip(LEVELS, per_level):
        df = load_hotpotqa_sample(level, num_samples=test_set_size + wanted)
        tail = df.iloc[test_set_size:].reset_index(drop=True)

        if len(tail) < wanted:
            print(
                f"[train] WARNING: only {len(tail)} '{level}' rows available "
                f"past index {test_set_size} (wanted {wanted})."
            )

        if len(tail) == 0:
            per_level_counts[level] = 0
            continue

        tail = tail.copy()
        tail["level"] = level
        pieces.append(tail)
        per_level_counts[level] = len(tail)

    if not pieces:
        raise RuntimeError(
            "No training rows could be collected -- check test_set_size "
            "vs. the HotpotQA pool size for each difficulty."
        )

    merged = (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return merged, per_level_counts


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
        f"[train] Building mixed training set "
        f"(num_train_samples={NUM_TRAIN_SAMPLES}, "
        f"test_set_size={TEST_SET_SIZE}, seed={SEED})..."
    )

    train_df, per_level_counts = build_training_set(
        test_set_size=TEST_SET_SIZE,
        num_train_samples=NUM_TRAIN_SAMPLES,
        seed=SEED,
    )

    print(
        f"[train] Collected {len(train_df)} rows "
        f"(per level: {per_level_counts})."
    )

    agent = MemoryAgent(frozen_rules_path=None)

    raw_rules: list[str] = []
    log_rows: list[dict] = []

    total_tokens = 0
    total_calls  = 0

    for index, row in train_df.iterrows():
        tools = make_hotpotqa_tools(row["context"])

        print(
            f"\n--- Train Q {index + 1}/{len(train_df)} "
            f"(level={row['level']}) ---"
        )
        print(f"Q: {row['question']}")

        result = agent.invoke(
            question=row["question"],
            tools=tools,
        )

        prediction    = result.get("answer", "")
        reference     = row["answer"]
        tokens_used   = result.get("total_tokens", 0)
        num_api_calls = result.get("num_api_calls", 0)
        tool_trace    = result.get("tool_trace", [])
        error         = result.get("error", "")

        em = exact_match(prediction, reference)
        f1 = f1_score_text(prediction, reference)

        total_tokens += tokens_used
        total_calls  += num_api_calls

        rule_added = False
        if f1 < REFLECTION_F1_THRESHOLD:
            rule, r_tokens, r_calls = agent.reflect(
                question=row["question"],
                tool_trace=tool_trace,
                prediction=prediction,
                reference=reference,
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

        log_rows.append(
            {
                "level":         row["level"],
                "prediction":    prediction,
                "reference":     reference,
                "f1":            f1,
                "em":            em,
                "tokens_used":   tokens_used,
                "num_api_calls": num_api_calls,
                "error":         error,
                "rule_added":    rule_added,
            }
        )
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        print(f"Pred: {prediction}")
        print(f"True: {reference}")
        print(f"EM: {em} | F1: {f1:.3f}")
        print(f"Tokens: {tokens_used} | API Calls: {num_api_calls}")
        if error:
            print(f"Error: {error}")

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
        "training_size":        len(train_df),
        "raw_rule_count":       raw_rule_count,
        "consolidator_top_n":   TOP_N,
        "per_level_counts":     per_level_counts,
        "reflection_threshold": REFLECTION_F1_THRESHOLD,
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
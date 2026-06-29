import pandas as pd
from pathlib import Path
import argparse


def error_count(df):
    """Number of rows with a non-empty `error` value (0 if no error column)."""
    if "error" not in df.columns:
        return 0
    err = df["error"]
    return int(err.notna().sum() - (err.astype(str).str.strip() == "").sum())


def aggregate(path):
    """Aggregate a single results CSV, auto-detecting the benchmark schema."""
    df = pd.read_csv(path)
    cols = df.columns

    summary = {"config": path.stem, "num_samples": len(df)}

    if "f1" in cols:  # HotpotQA schema
        token_col = "tokens_used"
        summary["avg_f1"] = df["f1"].mean()
        summary["avg_em"] = df["em"].mean()
    elif "reward" in cols:  # miniWOB schema
        token_col = "total_tokens"
        summary["avg_reward"] = df["reward"].mean()
        summary["success_rate"] = (df["reward"] > 0).mean()
        summary["avg_num_actions"] = df["num_actions"].mean()
    else:
        raise ValueError(
            f"Unrecognized schema in {path} (columns: {list(cols)})"
        )

    summary["total_tokens"] = df[token_col].sum()
    summary["avg_tokens_per_sample"] = df[token_col].mean()
    summary["total_api_calls"] = df["num_api_calls"].sum()
    summary["avg_api_calls"] = df["num_api_calls"].mean()
    summary["error_count"] = error_count(df)

    return summary


def print_kv(summary):
    """Original key/value output, used for the single --file case."""
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


# Canonical column order for the tidy CSV export. Configs from either benchmark
# are unioned; metrics absent for a given benchmark are left blank.
CSV_COLUMNS = [
    "benchmark", "arch", "difficulty", "config", "num_samples",
    "avg_f1", "avg_em", "avg_reward", "success_rate", "avg_num_actions",
    "total_tokens", "avg_tokens_per_sample", "total_api_calls",
    "avg_api_calls", "error_count",
]


def split_config(config):
    """Split a config stem like 'b1_hard' into ('b1', 'hard').

    Difficulty is the last underscore-separated token; everything before it is
    the architecture (so multi-token names like 'baseline' survive)."""
    arch, _, difficulty = config.rpartition("_")
    return (arch or config), difficulty


def write_csv(by_benchmark, out_path):
    """Write all aggregated rows to a single tidy long-format CSV."""
    rows = []
    for benchmark in sorted(by_benchmark):
        for summary in sorted(by_benchmark[benchmark], key=lambda r: r["config"]):
            arch, difficulty = split_config(summary["config"])
            rows.append({"benchmark": benchmark, "arch": arch,
                         "difficulty": difficulty, **summary})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=CSV_COLUMNS).round(4).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Single CSV file to aggregate (e.g. results/hotpotQA/c1_hard.csv). "
        "If omitted, all CSVs under --dir are aggregated into per-benchmark tables.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Root results directory to scan when --file is not given.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write the aggregated summary as a single tidy "
        "long-format CSV (for plotting). Does not affect console output.",
    )
    args = parser.parse_args()

    if args.file:
        print_kv(aggregate(Path(args.file)))
        return

    root = Path(args.dir)
    csvs = sorted(root.glob("*/*.csv"))
    if not csvs:
        print(f"No CSV files found under {root}")
        return

    # group by benchmark (parent folder) and print a table per benchmark
    by_benchmark = {}
    for csv in csvs:
        by_benchmark.setdefault(csv.parent.name, []).append(aggregate(csv))

    for benchmark in sorted(by_benchmark):
        rows = sorted(by_benchmark[benchmark], key=lambda r: r["config"])
        df = pd.DataFrame(rows).round(4)
        print(f"\n=== {benchmark} ===")
        print(df.to_string(index=False))

    if args.csv:
        write_csv(by_benchmark, args.csv)


if __name__ == "__main__":
    main()

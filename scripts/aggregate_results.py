import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="CSV file name from results folder to aggregate (e.g. results/hotpotQa/baseline_hard.csv)"
)
args = parser.parse_args()


path = Path(args.file)

df = pd.read_csv(path)

# basic aggregates
summary = {
    "num_samples": len(df),
    "avg_f1": df["f1"].mean(),
    "avg_em": df["em"].mean(),
    "total_tokens": df["tokens_used"].sum(),
    "avg_tokens_per_sample": df["tokens_used"].mean(),
    "total_api_calls": df["num_api_calls"].sum(),
    "avg_api_calls": df["num_api_calls"].mean()
}

for k, v in summary.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
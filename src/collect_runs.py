"""Builds an ablation table from finished training runs.

The spec csv lists the runs to include (columns: run, group, setting,
chosen); each run's config.json, log.csv and robust_eval json are read from
the run and eval directories. With --bench, throughput and peak memory from
src/bench.py are joined on arch (used for the backbone comparison).

Usage:
    python src/collect_runs.py --spec outputs/ablations/ablation_spec.csv \
        --out outputs/ablations/ablations.csv
    python src/collect_runs.py --spec outputs/ablations/backbone_spec.csv \
        --bench outputs/ablations/bench.json --out outputs/ablations/backbones.csv
"""

import argparse
import csv
import json
import os


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect(spec_row, runs_dir, eval_dir):
    run = spec_row["run"]
    cfg = read_json(os.path.join(runs_dir, run, "config.json"))
    summary = read_json(os.path.join(eval_dir, run + ".json"))["summary"]
    log = read_log(os.path.join(runs_dir, run, "log.csv"))
    best = max(log, key=lambda r: float(r["score"]))
    return {
        "group": spec_row["group"],
        "setting": spec_row["setting"],
        "run": run,
        "arch": cfg["arch"],
        "epochs": cfg["epochs"],
        "clean_auc": round(summary["clean_auc"], 4),
        "mean_auc": round(summary["mean_auc"], 4),
        "worst_auc": round(summary["worst_auc"], 4),
        "std_auc": round(summary["std_auc"], 4),
        "best_epoch": int(best["epoch"]) + 1,
        "train_seconds": round(sum(float(r["epoch_seconds"]) for r in log)),
        "chosen": spec_row.get("chosen", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--bench", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.spec, newline="", encoding="utf-8") as f:
        spec = list(csv.DictReader(f))
    rows = [collect(s, args.runs_dir, args.eval_dir) for s in spec]

    if args.bench:
        bench = {r["arch"]: r for r in read_json(args.bench)["results"]}
        for row in rows:
            b = bench[row["arch"]]
            row["img_per_s"] = b["img_per_s"]
            row["peak_mb"] = b["peak_mb"]
            row["params_m"] = b["params_m"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

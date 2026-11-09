"""Build soft targets for the confidently flagged inner-fold predictions."""

import csv
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv("data/splits/split_v1.csv")
    oof = read_csv("outputs/ablations/v6_clean_oof.csv")
    by_id = {row["image_id"]: row for row in oof}
    eligible = {row["image_id"] for row in rows
                if row["split"] == "train" and row["is_conflict"] != "1"}
    if len(by_id) != len(oof) or set(by_id) != eligible:
        raise ValueError("Inner-fold predictions must cover exactly the eligible training rows")

    relabelled = 0
    for row in rows:
        prediction = by_id.get(row["image_id"])
        if prediction is None:
            continue
        if prediction["label"] != row["label"]:
            raise ValueError(f"Label mismatch for {row['image_id']}")
        if prediction["flagged"] == "1":
            probability = float(prediction["p_fake"])
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid teacher probability for {row['image_id']}")
            # Keep one quarter of the original target to temper teacher errors.
            row["soft_label"] = f"{0.25 * int(row['label']) + 0.75 * probability:.6f}"
            relabelled += 1

    out = Path("data/splits/split_v1_soft_relabel.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Softened {relabelled} flagged training targets in {out}")


if __name__ == "__main__":
    main()

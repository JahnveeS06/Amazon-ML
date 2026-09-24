"""
make_split.py  --  run ONCE, then share split_assignments.tsv with the team.
 
Grouped + stratified 70/15/15 split of the TRAIN folder only.
  * group  = one Source 1 entity + all its matched S2/S3 records
  * strata = country x match-count bucket (0, 1, 2+)
  * orphan S2/S3 records (in no ground-truth row) are spread proportionally
    by source and country so validation pools contain realistic distractors
 
Usage:
    python make_split.py --train-dir dataset/train --out split_assignments.tsv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
SEED = 42
FRACS = {"train": 0.70, "val": 0.15, "holdout": 0.15}
 
 
def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
 
 
def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2+"
 
 
def main(train_dir, out_path):
    s1 = read_tsv(f"{train_dir}/train_source1.tsv")
    s2 = read_tsv(f"{train_dir}/train_source2.tsv")
    s3 = read_tsv(f"{train_dir}/train_source3.tsv")
    gt = read_tsv(f"{train_dir}/train_ground_truth.tsv")
 
    # ground truth: S1 id -> list of matched S2/S3 ids (empty string = singleton)
    gt["matches"] = gt["matched_entity_ids"].apply(
        lambda x: [i.strip() for i in x.split(",") if i.strip()]
    )
    gt = gt.set_index("source1_entity_id")
 
    # sanity check: an S2/S3 id should not belong to two different S1 entities
    seen = {}
    clashes = 0
    for s1_id, row in gt.iterrows():
        for m in row["matches"]:
            if m in seen and seen[m] != s1_id:
                clashes += 1
            seen[m] = s1_id
    if clashes:
        print(f"WARNING: {clashes} S2/S3 ids are matched to more than one S1 entity. "
              "Tell the team before using this split.")
 
    # ---- stage 1: split S1 entities (each S1 entity is its own group) ----
    ent = s1[["entity_id", "country"]].copy()
    ent["n_matches"] = ent["entity_id"].map(lambda i: len(gt.loc[i, "matches"]) if i in gt.index else 0)
    ent["stratum"] = ent["country"] + "|" + ent["n_matches"].map(bucket)
    # merge very rare strata so stratification does not crash
    counts = ent["stratum"].value_counts()
    ent.loc[ent["stratum"].map(counts) < 4, "stratum"] = "rare"
    strat = ent["stratum"] if (ent["stratum"].value_counts() >= 4).all() else None
 
    train_ids, rest_ids = train_test_split(
        ent["entity_id"], train_size=FRACS["train"], random_state=SEED, stratify=strat
    )
    rest = ent[ent["entity_id"].isin(rest_ids)]
    rest_strat = rest["stratum"] if (rest["stratum"].value_counts() >= 2).all() else None
    val_ids, hold_ids = train_test_split(
        rest["entity_id"], train_size=0.5, random_state=SEED, stratify=rest_strat
    )
    split_of_s1 = {**{i: "train" for i in train_ids},
                   **{i: "val" for i in val_ids},
                   **{i: "holdout" for i in hold_ids}}
 
    # ---- stage 2: every matched S2/S3 record inherits its S1 entity's split ----
    rows = []
    for _, r in s1.iterrows():
        rows.append((r.entity_id, "S1", r.country, split_of_s1[r.entity_id], "reference"))
    matched_split = {m: split_of_s1[s1_id] for m, s1_id in seen.items() if s1_id in split_of_s1}
    rng = np.random.default_rng(SEED)
    for src, df in (("S2", s2), ("S3", s3)):
        for country, g in df.groupby("country"):
            orphan_mask = ~g["entity_id"].isin(matched_split)
            for _, r in g[~orphan_mask].iterrows():
                rows.append((r.entity_id, src, country, matched_split[r.entity_id], "matched"))
            # ---- stage 3: orphans spread proportionally per source and country ----
            orphans = g[orphan_mask]
            draw = rng.choice(list(FRACS), size=len(orphans), p=list(FRACS.values()))
            for eid, sp in zip(orphans["entity_id"], draw):
                rows.append((eid, src, country, sp, "orphan"))
 
    out = pd.DataFrame(rows, columns=["entity_id", "source", "country", "split", "role"])
    out.to_csv(out_path, sep="\t", index=False)
 
    print(f"Saved {len(out)} rows to {out_path}\n")
    print("Records per split x source:")
    print(pd.crosstab(out["split"], out["source"]), "\n")
    s1_view = out[out["source"] == "S1"].merge(ent[["entity_id", "n_matches"]], on="entity_id")
    s1_view["bucket"] = s1_view["n_matches"].map(bucket)
    print("S1 entities per split x country x match bucket (should be proportional):")
    print(pd.crosstab([s1_view["country"], s1_view["bucket"]], s1_view["split"]))
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--out", default="split_assignments.tsv")
    a = ap.parse_args()
    main(a.train_dir, a.out)
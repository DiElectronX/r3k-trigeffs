#!/usr/bin/env python3
import glob
import json
import os
from pathlib import Path

def load_brilcalc_csv(csv_path: str):
    """
    Reads brilcalc --byls --output-style csv output.
    Returns: dict[(run:int, ls:int)] = recorded_lumi(float)
    """
    lumi = {}
    header = None
    rec_col = None

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # brilcalc often prefixes header/comments with '#'
            if line.startswith("#"):
                maybe = line.lstrip("#").strip()
                if maybe.startswith("run:fill"):
                    header = [h.strip() for h in maybe.split(",")]
                    rec_col = next((h for h in header if "recorded" in h), None)
                continue

            # sometimes header is not commented
            if header is None and line.startswith("run:fill"):
                header = [h.strip() for h in line.split(",")]
                rec_col = next((h for h in header if "recorded" in h), None)
                continue

            if header is None or rec_col is None:
                raise RuntimeError("Could not find brilcalc CSV header or recorded column")

            fields = [x.strip() for x in line.split(",")]
            row = dict(zip(header, fields))

            run = int(row["run:fill"].split(":")[0])

            # ls column is often like "123:123" -> take first number
            ls = int(row["ls"].split(":")[0])

            recorded = float(row[rec_col])
            lumi[(run, ls)] = recorded

    return lumi

def iter_run_ls(mask_dict):
    for run, ranges in mask_dict.items():
        run = int(run)
        for lo, hi in ranges:
            lo = int(lo); hi = int(hi)
            for ls in range(lo, hi + 1):
                yield run, ls

def lumi_for_json(mask_path: str, lumi_map):
    with open(mask_path) as f:
        mask = json.load(f)
    total = 0.0
    missing = 0
    for key in iter_run_ls(mask):
        if key in lumi_map:
            total += lumi_map[key]
        else:
            missing += 1
    return total, missing

def main():
    csv_path = "lumi_OR_byLS.csv"
    or_json = "trigger_OR.json"
    excl_glob = "*_Excl_Final.json"   # or *_Excl_Golden.json if that's your choice

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Create it with brilcalc --byls --output-style csv first."
        )

    lumi_map = load_brilcalc_csv(csv_path)

    L_or, miss_or = lumi_for_json(or_json, lumi_map)
    print(f"OR lumi (recorded): {L_or:.6f}  | missing LS rows in CSV lookup: {miss_or}")

    total_excl = 0.0
    weights = []

    for fn in sorted(glob.glob(excl_glob)):
        L_i, miss_i = lumi_for_json(fn, lumi_map)
        w_i = (L_i / L_or) if L_or > 0 else 0.0
        weights.append((Path(fn).name, L_i, w_i, miss_i))
        total_excl += L_i

    for name, L_i, w_i, miss_i in weights:
        print(f"{name:30s}  L={L_i:12.6f}  w={w_i:10.6f}  missingLS={miss_i}")

    sumw = sum(w for _, _, w, _ in weights)
    coverage = total_excl / L_or if L_or > 0 else 0.0
    print(f"\nSum of weights: {sumw:.6f}")
    print(f"Coverage (sum L_excl / L_OR): {coverage:.6f}")
    print(f"Missing lumi fraction (1-coverage): {1.0 - coverage:.6f}")

if __name__ == "__main__":
    main()

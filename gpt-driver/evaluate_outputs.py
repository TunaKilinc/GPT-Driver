import argparse
import json
import re
import ast
from collections import defaultdict

import numpy as np


def _extract_first_bracket_block(text: str):
    """Return first [...] block in text (non-greedy)."""
    m = re.search(r"\[[\s\S]*?\]", text)
    return m.group(0) if m else None


def parse_gpt_traj(gpt_text: str):
    """Parse GPT trajectory from GPT field. Returns np.ndarray (N,2) or None."""
    block = _extract_first_bracket_block(gpt_text)
    if not block:
        return None
    try:
        traj = ast.literal_eval(block)
        arr = np.array(traj, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    # If model returned 7 points and the first is (0,0), drop it (common formatting issue)
    if arr.shape[0] == 7 and np.linalg.norm(arr[0]) < 1e-6:
        arr = arr[1:]

    return arr


def parse_gt_traj(gt_text: str):
    """Parse GT trajectory from GT field. Returns np.ndarray (N,2) or None."""
    # Usually GT contains "Trajectory:\n[...]"
    part = gt_text.split("Trajectory")[-1] if "Trajectory" in gt_text else gt_text
    block = _extract_first_bracket_block(part)
    if not block:
        return None
    try:
        traj = ast.literal_eval(block)
        arr = np.array(traj, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def parse_meta_action(gt_text: str):
    m = re.search(r"Meta Action:\s*(.*)", gt_text)
    return m.group(1).strip() if m else "UNKNOWN"


def summarize(name: str, values: np.ndarray):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        print(f"{name}: (no data)")
        return
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    print(
        f"{name}: mean={values.mean():.3f} | median={p50:.3f} | p25={p25:.3f} | p75={p75:.3f} | "
        f"min={values.min():.3f} | max={values.max():.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to *_temp.jsonl file")
    ap.add_argument("--max_n", type=int, default=0, help="If >0, evaluate only first N lines")
    args = ap.parse_args()

    ade_list = []
    fde_list = []
    parse_fail = 0

    by_action_ade = defaultdict(list)
    by_action_fde = defaultdict(list)

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_n > 0 and i >= args.max_n:
                break
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)
            token = d.get("token", "UNKNOWN_TOKEN")
            gpt = d.get("GPT", "")
            gt = d.get("GT", "")

            gpt_arr = parse_gpt_traj(gpt)
            gt_arr = parse_gt_traj(gt)
            if gpt_arr is None or gt_arr is None:
                parse_fail += 1
                continue

            # Align by minimum length (normally both are 6 points)
            n = min(len(gpt_arr), len(gt_arr))
            if n == 0:
                parse_fail += 1
                continue

            diff = np.linalg.norm(gpt_arr[:n] - gt_arr[:n], axis=1)
            ade = float(diff.mean())
            fde = float(diff[-1])

            ade_list.append(ade)
            fde_list.append(fde)

            action = parse_meta_action(gt)
            by_action_ade[action].append(ade)
            by_action_fde[action].append(fde)

    ade_arr = np.array(ade_list, dtype=float)
    fde_arr = np.array(fde_list, dtype=float)

    print(f"Evaluated samples: {len(ade_list)}")
    print(f"Parse failures: {parse_fail}")
    summarize("ADE (Average Displacement Error; Ortalama Konum Hatası) [m]", ade_arr)
    summarize("FDE (Final Displacement Error; Final Konum Hatası) [m]", fde_arr)

    # Threshold success rates
    for th in [1, 2, 5, 10, 20]:
        rate = float((fde_arr <= th).mean()) if fde_arr.size else 0.0
        print(f"FDE <= {th} m rate: {rate*100:.2f}%")

    # Top actions by count
    print("\nTop Meta Action groups (by count):")
    groups = sorted(by_action_ade.keys(), key=lambda k: len(by_action_ade[k]), reverse=True)
    for k in groups[:10]:
        a = np.array(by_action_ade[k], dtype=float)
        f = np.array(by_action_fde[k], dtype=float)
        print(
            f"- {k}: n={len(a)} | ADE_mean={a.mean():.3f} | FDE_mean={f.mean():.3f} | "
            f"ADE_median={np.median(a):.3f} | FDE_median={np.median(f):.3f}"
        )


if __name__ == "__main__":
    main()

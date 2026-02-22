import argparse
import json
import re
import ast
from collections import defaultdict

import numpy as np


def _extract_first_bracket_block(text: str):
    m = re.search(r"\[[\s\S]*?\]", text)
    return m.group(0) if m else None


def parse_gpt_traj(gpt_text: str):
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

    # If 7 points and first is (0,0), drop it
    if arr.shape[0] == 7 and np.linalg.norm(arr[0]) < 1e-6:
        arr = arr[1:]

    return arr


def parse_gt_traj(gt_text: str):
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


def main():
    ap = argparse.ArgumentParser(description="Analyze worst-case failures from GPT-Driver temp log.")
    ap.add_argument("--jsonl", required=True, help="Path to *_temp.jsonl file")
    ap.add_argument("--topk", type=int, default=30, help="How many worst samples to print (by FDE)")
    ap.add_argument("--max_n", type=int, default=0, help="If >0, analyze only first N lines")
    args = ap.parse_args()

    rows = []
    parse_fail = 0

    # Per-action buckets (store (fde, ade, token))
    action_bucket = defaultdict(list)

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

            n = min(len(gpt_arr), len(gt_arr))
            if n == 0:
                parse_fail += 1
                continue

            diff = np.linalg.norm(gpt_arr[:n] - gt_arr[:n], axis=1)
            ade = float(diff.mean())
            fde = float(diff[-1])

            # Simple extra diagnostics
            # scale ratio: compare GT final distance vs GPT final distance from origin
            gt_final = float(np.linalg.norm(gt_arr[n - 1]))
            gpt_final = float(np.linalg.norm(gpt_arr[n - 1]))
            scale_ratio = (gpt_final / gt_final) if gt_final > 1e-6 else np.nan

            action = parse_meta_action(gt)

            rows.append({
                "token": token,
                "action": action,
                "ade": ade,
                "fde": fde,
                "scale_ratio": scale_ratio,
                "gpt_last": tuple(map(float, gpt_arr[n - 1])),
                "gt_last": tuple(map(float, gt_arr[n - 1])),
                "gpt_first": tuple(map(float, gpt_arr[0])),
                "gt_first": tuple(map(float, gt_arr[0])),
            })

            action_bucket[action].append((fde, ade, token, scale_ratio))

    print(f"Parsed samples: {len(rows)}")
    print(f"Parse failures: {parse_fail}")

    if not rows:
        print("No parsed rows. Check jsonl path and format.")
        return

    # Sort by FDE desc
    rows_sorted = sorted(rows, key=lambda r: r["fde"], reverse=True)

    print("\n=== TOP WORST SAMPLES (by FDE) ===")
    for r in rows_sorted[:args.topk]:
        print(
            f"- token={r['token']} | action={r['action']} | "
            f"FDE={r['fde']:.3f}m | ADE={r['ade']:.3f}m | scale_ratio={r['scale_ratio']:.3f} | "
            f"GPT_last={r['gpt_last']} | GT_last={r['gt_last']}"
        )

    # Show action-level worst cases (top 5 actions by mean FDE)
    print("\n=== ACTION SUMMARY (sorted by mean FDE) ===")
    action_stats = []
    for action, vals in action_bucket.items():
        fdes = np.array([v[0] for v in vals], dtype=float)
        ades = np.array([v[1] for v in vals], dtype=float)
        ratios = np.array([v[3] for v in vals if not np.isnan(v[3])], dtype=float)
        action_stats.append((
            action,
            len(vals),
            float(fdes.mean()),
            float(np.median(fdes)),
            float(ades.mean()),
            float(np.median(ades)),
            float(np.nanmean(ratios)) if ratios.size else np.nan,
        ))

    action_stats.sort(key=lambda x: x[2], reverse=True)  # by mean FDE
    for action, n, fde_mean, fde_med, ade_mean, ade_med, ratio_mean in action_stats[:15]:
        print(
            f"- {action}: n={n} | FDE_mean={fde_mean:.3f} | FDE_median={fde_med:.3f} | "
            f"ADE_mean={ade_mean:.3f} | ADE_median={ade_med:.3f} | avg_scale_ratio={ratio_mean:.3f}"
        )

    # Per-action top-k worst tokens
    print("\n=== WORST TOKENS PER ACTION (top 5 by FDE) ===")
    for action, vals in sorted(action_bucket.items(), key=lambda kv: np.mean([v[0] for v in kv[1]]), reverse=True)[:10]:
        vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)[:5]
        print(f"\nAction: {action}")
        for fde, ade, token, ratio in vals_sorted:
            print(f"  - token={token} | FDE={fde:.3f} | ADE={ade:.3f} | scale_ratio={ratio:.3f}")

    # Optional: simple histogram bins for FDE
    fde_all = np.array([r["fde"] for r in rows], dtype=float)
    bins = [0, 2, 5, 10, 20, 50, 1e9]
    counts = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        counts.append(int(((fde_all >= b0) & (fde_all < b1)).sum()))
    total = len(fde_all)
    print("\n=== FDE DISTRIBUTION (counts) ===")
    for (b0, b1), c in zip(zip(bins[:-1], bins[1:]), counts):
        pct = 100.0 * c / total if total else 0.0
        label = f"[{b0},{b1})" if b1 < 1e9 else f"[{b0},inf)"
        print(f"- {label} m: {c} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

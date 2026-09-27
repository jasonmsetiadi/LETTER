"""Compare all collaborative / popularity signals on a dataset."""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# Add RQ-VAE directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from truncate_indices import (
    compute_interaction_signals,
    truncate_indices,
)


def compute_spearman_correlation(score_dict_a, score_dict_b, items):
    """Compute Spearman rank correlation between two score dictionaries."""
    n = len(items)
    if n <= 1:
        return 1.0

    sorted_a = sorted(items, key=lambda k: (score_dict_a.get(str(k), 0.0), str(k)))
    rank_a = {k: r for r, k in enumerate(sorted_a)}

    sorted_b = sorted(items, key=lambda k: (score_dict_b.get(str(k), 0.0), str(k)))
    rank_b = {k: r for r, k in enumerate(sorted_b)}

    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in items)
    rho = 1.0 - (6.0 * d2) / (n * (n * n - 1))
    return float(rho)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare all collaborative / popularity signals for variable-length Semantic IDs."
    )
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset parent directory (defaults to <repo-root>/data).",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Input fixed-length .index.json file path.",
    )
    parser.add_argument(
        "--inter-file",
        type=str,
        default=None,
        help="Interaction .inter.json file path.",
    )
    parser.add_argument(
        "--cf-emb-file",
        type=str,
        default=None,
        help="Optional path to CF embeddings (.pt, .npy, .json) for cf_density signal.",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default="frequency,user_entropy,pagerank,target,composite,cf_density",
        help="Comma-separated list of signals to compare (default: all).",
    )
    parser.add_argument("--min-length", type=int, default=1, help="Minimum SID length.")
    parser.add_argument("--max-length", type=int, default=4, help="Maximum SID length.")
    parser.add_argument(
        "--save-indices",
        action="store_true",
        default=True,
        help="Save generated variable-length indices for each signal (default: True).",
    )
    parser.add_argument(
        "--no-save-indices",
        action="store_false",
        dest="save_indices",
        help="Do not write index files to disk.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save consolidated comparison JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Path to save consolidated markdown table report.",
    )
    return parser.parse_args()


SIGNAL_METADATA = {
    "frequency": {
        "label": "Raw Frequency",
        "basis": "Unigram count sum_u sum_t I(s_ut = i)",
        "file_tag": "pop",
    },
    "user_entropy": {
        "label": "User Audience Entropy",
        "basis": "Shannon entropy H(i) over user audience",
        "file_tag": "pop-entropy",
    },
    "pagerank": {
        "label": "Sequential PageRank",
        "basis": "Stationary distribution on transition graph i -> j",
        "file_tag": "pop-pagerank",
    },
    "target": {
        "label": "Target-Position Frequency",
        "basis": "Frequency in sequence-final target position",
        "file_tag": "pop-target",
    },
    "composite": {
        "label": "Composite Score",
        "basis": "log2(1 + f_i) * (1 + H_user(i))",
        "file_tag": "pop-composite",
    },
    "cf_density": {
        "label": "CF Manifold Isolation",
        "basis": "1 - kNN crowd cosine density in CF space",
        "file_tag": "pop-cf",
    },
}


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root) if args.data_root else repo_root / "data"
    dataset_dir = data_root / args.dataset

    index_file = (
        Path(args.index_file)
        if args.index_file
        else dataset_dir / f"{args.dataset}.index.json"
    )
    if not index_file.exists():
        # Fallback to fixed-for-varlen
        alt_index = dataset_dir / f"{args.dataset}.index.fixed-for-varlen.json"
        if alt_index.exists():
            index_file = alt_index
        else:
            sys.exit(f"Error: Index file not found at {index_file} or {alt_index}")

    inter_file = (
        Path(args.inter_file)
        if args.inter_file
        else dataset_dir / f"{args.dataset}.inter.json"
    )
    if not inter_file.exists():
        sys.exit(f"Error: Interaction file not found at {inter_file}")

    cf_emb_file = args.cf_emb_file
    if not cf_emb_file:
        candidate_cf = repo_root / "RQ-VAE" / "ckpt" / f"{args.dataset}-32d-sasrec.pt"
        if candidate_cf.exists():
            cf_emb_file = str(candidate_cf)

    with index_file.open(encoding="utf-8") as f:
        indices = json.load(f)

    # Filter signals to compare
    requested_signals = [s.strip() for s in args.signals.replace(" ", ",").split(",") if s.strip()]
    signals_to_run = []
    for s in requested_signals:
        if s == "cf_density" and not cf_emb_file:
            print(f"[Notice] Skipping '{s}': CF embedding file not specified or found.")
            continue
        if s == "cf_density":
            # Check if torch or numpy is available
            try:
                import torch
            except ImportError:
                try:
                    import numpy
                except ImportError:
                    print(f"[Notice] Skipping '{s}': Requires PyTorch or NumPy installed.")
                    continue
        if s in SIGNAL_METADATA:
            signals_to_run.append(s)
        else:
            print(f"[Warning] Unknown signal '{s}', skipping.")

    if not signals_to_run:
        sys.exit("Error: No valid signals selected to run.")

    # Compute raw frequency scores once as ground truth for correlations and traffic weighting
    raw_scores, raw_freqs = compute_interaction_signals(inter_file, signal="frequency")
    items = list(indices.keys())
    total_traffic = max(1, sum(raw_freqs.get(str(i), 0) for i in items))

    # Evaluate each signal
    results = []
    all_scores = {"frequency": raw_scores}

    print("\n" + "=" * 90)
    print(f" LETTER Collaborative Signal Comparison for Dataset: {args.dataset}")
    print(f" Catalog Items: {len(indices):,} | Total Interactions: {sum(raw_freqs.values()):,}")
    print(f" Length Range: [{args.min_length}, {args.max_length}]")
    print("=" * 90)

    for sig in signals_to_run:
        meta = SIGNAL_METADATA.get(sig, {"label": sig, "basis": "", "file_tag": sig})
        scores, _ = compute_interaction_signals(
            inter_file,
            signal=sig,
            cf_emb_file=cf_emb_file if sig in ("cf_density", "composite") else None,
        )
        all_scores[sig] = scores

        truncated, lengths = truncate_indices(
            indices,
            min_length=args.min_length,
            max_length=args.max_length,
            strategy="collaborative",
            item_scores=scores,
            item_frequencies=raw_freqs,
        )

        mean_len = sum(lengths.values()) / len(lengths)
        traffic_w_len = sum(lengths[i] * raw_freqs.get(str(i), 0) for i in truncated) / total_traffic
        token_savings_pct = (1.0 - traffic_w_len / args.max_length) * 100.0

        length_dist = dict(sorted(Counter(lengths.values()).items()))
        unique_ids = len({tuple(v) for v in truncated.values()})
        collisions = len(truncated) - unique_ids

        rho = compute_spearman_correlation(raw_scores, scores, items)

        res_entry = {
            "signal": sig,
            "label": meta["label"],
            "basis": meta["basis"],
            "file_tag": meta["file_tag"],
            "mean_length": mean_len,
            "traffic_weighted_length": traffic_w_len,
            "token_savings_pct": token_savings_pct,
            "spearman_rho_vs_frequency": rho,
            "length_distribution": length_dist,
            "unique_ids": unique_ids,
            "collisions": collisions,
        }
        results.append(res_entry)

        if args.save_indices:
            file_tag = meta["file_tag"]
            suffix = f".{file_tag}"
            tag_suffix = "" if (args.max_length == 4 and args.min_length == 1) else f".max{args.max_length}"
            out_name = f"{args.dataset}.index.varlen{suffix}{tag_suffix}.json"
            out_path = dataset_dir / out_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(truncated, f)

            summary_path = out_path.with_suffix(".summary.json")
            summary_payload = {
                "input": str(index_file),
                "strategy": "collaborative",
                "collab_signal": sig,
                "items": len(truncated),
                "min_length": min(lengths.values()),
                "max_length": max(lengths.values()),
                "mean_length": mean_len,
                "traffic_weighted_mean_length": traffic_w_len,
                "weighted_mean_length": traffic_w_len,
                "signal_weighted_mean_length": sum(lengths[i] * scores.get(str(i), 0.0) for i in truncated) / max(1e-9, sum(abs(scores.get(str(i), 0.0)) for i in truncated)),
                "spearman_rho_vs_frequency": rho,
                "length_distribution": length_dist,
                "unique_ids": unique_ids,
                "collisions": collisions,
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)

    # Print Comparison Table
    header = (
        f"| {'Signal':<20} | {'Catalog Mean':<12} | {'Traffic W-Len':<14} | "
        f"{'Token Savings':<13} | {'L=1':<5} | {'L=2':<6} | {'L=3':<6} | {'L=4':<6} | {'Spearman ρ':<10} |"
    )
    separator = (
        f"|{'-'*22}|{'-'*14}|{'-'*16}|{'-'*15}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*12}|"
    )

    print("\n" + header)
    print(separator)
    for r in results:
        d = r["length_distribution"]
        l1 = d.get(1, 0)
        l2 = d.get(2, 0)
        l3 = d.get(3, 0)
        l4 = d.get(4, 0)
        print(
            f"| {r['label']:<20} | {r['mean_length']:12.3f} | {r['traffic_weighted_length']:14.3f} | "
            f"{r['token_savings_pct']:11.1f}% | {l1:<5} | {l2:<6} | {l3:<6} | {l4:<6} | {r['spearman_rho_vs_frequency']:10.4f} |"
        )
    print(separator)

    # Save Markdown output if requested
    if args.output_markdown:
        md_lines = [
            f"# LETTER Collaborative Signal Comparison Report: {args.dataset}",
            "",
            f"- **Dataset**: `{args.dataset}`",
            f"- **Items**: {len(indices):,}",
            f"- **Fixed ID Depth**: {args.max_length} tokens",
            f"- **Minimum Allowable Depth**: {args.min_length} token",
            "",
            "## Summary Table",
            "",
            header,
            separator,
        ]
        for r in results:
            d = r["length_distribution"]
            l1 = d.get(1, 0)
            l2 = d.get(2, 0)
            l3 = d.get(3, 0)
            l4 = d.get(4, 0)
            md_lines.append(
                f"| {r['label']:<20} | {r['mean_length']:12.3f} | {r['traffic_weighted_length']:14.3f} | "
                f"{r['token_savings_pct']:11.1f}% | {l1:<5} | {l2:<6} | {l3:<6} | {l4:<6} | {r['spearman_rho_vs_frequency']:10.4f} |"
            )
        md_lines.append(separator)
        md_lines.append("")
        md_lines.append("## Signal Definitions & Methodologies")
        md_lines.append("")
        for r in results:
            md_lines.append(f"- **{r['label']}** (`{r['signal']}`): {r['basis']}")
        md_path = Path(args.output_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved Markdown report to: {md_path}")

    # Save JSON output if requested
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "results": results}, f, indent=2)
        print(f"Saved JSON comparison data to: {json_path}")


if __name__ == "__main__":
    main()

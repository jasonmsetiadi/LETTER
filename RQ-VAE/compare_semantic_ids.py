"""Compare Semantic ID generation strategies (fixed, shortest_unique, residual, collaborative) on CPU."""

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
        description="CPU-only evaluation comparing generated Semantic IDs across strategies (fixed, shortest_unique, residual, collaborative)."
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
        "--include-baselines",
        action="store_true",
        default=False,
        help="Include Fixed (L=max_length) and Shortest Unique prefix baselines in comparison.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        default=False,
        help="Compute and display pairwise Semantic ID exact match rate and length divergence matrix across strategies.",
    )
    parser.add_argument(
        "--residuals-file",
        type=str,
        default=None,
        help="Optional residuals JSON file to include residual truncation strategy.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.2,
        help="Cumulative reconstruction error threshold for residual strategy (default: 0.2).",
    )
    parser.add_argument(
        "--save-indices",
        action="store_true",
        default=False,
        help="Also write generated variable-length index JSON files to disk (default: False).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save report files (defaults to <data-root>/<dataset>).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save consolidated comparison JSON (defaults to <output-dir>/semantic_id_comparison.json).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Path to save consolidated markdown table report (defaults to <output-dir>/semantic_id_comparison.md).",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        default=False,
        help="Do not save comparison report files (JSON/Markdown) to disk; only print to console.",
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
    strategy_lengths = {}
    all_scores = {"frequency": raw_scores}

    print("\n" + "=" * 90)
    print(f" LETTER Collaborative Signal Comparison for Dataset: {args.dataset}")
    print(f" Catalog Items: {len(indices):,} | Total Interactions: {sum(raw_freqs.values()):,}")
    print(f" Length Range: [{args.min_length}, {args.max_length}]")
    print("=" * 90)

    if args.include_baselines:
        # Fixed-length baseline
        strategy_lengths["fixed"] = {str(i): args.max_length for i in items}
        results.append({
            "signal": "fixed",
            "label": f"Fixed (L={args.max_length})",
            "basis": f"Uniform fixed codebook depth L={args.max_length}",
            "file_tag": "fixed",
            "mean_length": float(args.max_length),
            "traffic_weighted_length": float(args.max_length),
            "token_savings_pct": 0.0,
            "spearman_rho_vs_frequency": None,
            "length_distribution": {args.max_length: len(indices)},
            "unique_ids": len(indices),
            "collisions": 0,
        })

        # Shortest unique prefix baseline
        trunc_su, lens_su = truncate_indices(
            indices,
            min_length=args.min_length,
            max_length=args.max_length,
            strategy="shortest_unique",
        )
        strategy_lengths["shortest_unique"] = {str(k): v for k, v in lens_su.items()}
        su_mean = sum(lens_su.values()) / len(lens_su)
        su_traffic_w = sum(lens_su[i] * raw_freqs.get(str(i), 0) for i in trunc_su) / total_traffic
        su_savings = (1.0 - su_traffic_w / args.max_length) * 100.0
        su_dist = dict(sorted(Counter(lens_su.values()).items()))
        su_uniq = len({tuple(v) for v in trunc_su.values()})
        su_coll = len(trunc_su) - su_uniq
        results.append({
            "signal": "shortest_unique",
            "label": "Shortest Unique",
            "basis": "Shortest unambiguous trie prefix per item",
            "file_tag": "shortest_unique",
            "mean_length": su_mean,
            "traffic_weighted_length": su_traffic_w,
            "token_savings_pct": su_savings,
            "spearman_rho_vs_frequency": None,
            "length_distribution": su_dist,
            "unique_ids": su_uniq,
            "collisions": su_coll,
        })

        # Residual strategy (if residuals file is available)
        res_file_cand = (
            Path(args.residuals_file)
            if args.residuals_file
            else dataset_dir / f"{args.dataset}.residuals.json"
        )
        if res_file_cand.exists():
            with res_file_cand.open(encoding="utf-8") as rf:
                loaded_residuals = json.load(rf)
            trunc_res, lens_res = truncate_indices(
                indices,
                min_length=args.min_length,
                max_length=args.max_length,
                strategy="residual",
                residuals=loaded_residuals,
                residual_threshold=args.residual_threshold,
            )
            strategy_lengths["residual"] = {str(k): v for k, v in lens_res.items()}
            res_mean = sum(lens_res.values()) / len(lens_res)
            res_traffic_w = sum(lens_res[i] * raw_freqs.get(str(i), 0) for i in trunc_res) / total_traffic
            res_savings = (1.0 - res_traffic_w / args.max_length) * 100.0
            res_dist = dict(sorted(Counter(lens_res.values()).items()))
            res_uniq = len({tuple(v) for v in trunc_res.values()})
            res_coll = len(trunc_res) - res_uniq
            results.append({
                "signal": "residual",
                "label": f"Residual (Thresh={args.residual_threshold})",
                "basis": f"Shortest prefix with cumulative reconstruction error <= {args.residual_threshold}",
                "file_tag": "res",
                "mean_length": res_mean,
                "traffic_weighted_length": res_traffic_w,
                "token_savings_pct": res_savings,
                "spearman_rho_vs_frequency": None,
                "length_distribution": res_dist,
                "unique_ids": res_uniq,
                "collisions": res_coll,
            })

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
        strategy_lengths[sig] = {str(k): v for k, v in lengths.items()}

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
        rho_val = r["spearman_rho_vs_frequency"]
        rho_str = f"{rho_val:10.4f}" if rho_val is not None else "         -"
        print(
            f"| {r['label']:<20} | {r['mean_length']:12.3f} | {r['traffic_weighted_length']:14.3f} | "
            f"{r['token_savings_pct']:11.1f}% | {l1:<5} | {l2:<6} | {l3:<6} | {l4:<6} | {rho_str} |"
        )
    print(separator)

    pairwise_match = {}
    pairwise_mae = {}
    pairwise_md_sections = []
    if args.pairwise and len(results) > 1:
        strat_keys = [r["signal"] for r in results]
        strat_labels = [r["label"] for r in results]
        n_strats = len(strat_keys)

        for s1 in strat_keys:
            pairwise_match[s1] = {}
            pairwise_mae[s1] = {}
            for s2 in strat_keys:
                l1 = strategy_lengths[s1]
                l2 = strategy_lengths[s2]
                exact_matches = sum(l1.get(str(i), 0) == l2.get(str(i), 0) for i in items)
                match_pct = (exact_matches / len(items)) * 100.0
                mae = sum(abs(l1.get(str(i), 0) - l2.get(str(i), 0)) for i in items) / len(items)
                pairwise_match[s1][s2] = round(match_pct, 2)
                pairwise_mae[s1][s2] = round(mae, 3)

        max_label_w = max(len(l) for l in strat_labels)
        col_w = max(14, max_label_w + 2)
        row_fmt = f"| {{:<{col_w}}} | " + " | ".join([f"{{:<{col_w}}}" for _ in range(n_strats)]) + " |"
        sep_fmt = f"|{'-' * (col_w + 2)}|" + "|".join([f"{'-' * (col_w + 2)}" for _ in range(n_strats)]) + "|"

        print("\n" + "=" * 90)
        print(" Pairwise Exact Semantic ID Agreement Rate (%)")
        print(" (Percentage of items assigned identical Semantic IDs between strategies)")
        print("=" * 90)
        print(row_fmt.format("Strategy", *strat_labels))
        print(sep_fmt)
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in strat_keys]
            print(row_fmt.format(l1, *vals))
        print(sep_fmt)

        print("\n" + "=" * 90)
        print(" Pairwise Mean Absolute Length Difference (Tokens)")
        print(" (Average token length divergence per item: sum(|L_A - L_B|) / N)")
        print("=" * 90)
        print(row_fmt.format("Strategy", *strat_labels))
        print(sep_fmt)
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in strat_keys]
            print(row_fmt.format(l1, *vals))
        print(sep_fmt)

        pairwise_md_sections = [
            "## Pairwise Exact Semantic ID Agreement Rate (%)",
            "",
            "> Percentage of items in the catalog that receive an identical Semantic ID across strategies.",
            "",
            row_fmt.format("Strategy", *strat_labels),
            sep_fmt,
        ]
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in strat_keys]
            pairwise_md_sections.append(row_fmt.format(l1, *vals))
        pairwise_md_sections.append(sep_fmt)
        pairwise_md_sections.append("")

        pairwise_md_sections.extend([
            "## Pairwise Mean Absolute Length Difference (Tokens)",
            "",
            "> Average token length difference per item (|L_A - L_B|). Lower value indicates closer length profiles.",
            "",
            row_fmt.format("Strategy", *strat_labels),
            sep_fmt,
        ])
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in strat_keys]
            pairwise_md_sections.append(row_fmt.format(l1, *vals))
        pairwise_md_sections.append(sep_fmt)
        pairwise_md_sections.append("")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    default_md = out_dir / "semantic_id_comparison.md"
    default_json = out_dir / "semantic_id_comparison.json"

    out_md = None if args.no_save_report else (Path(args.output_markdown) if args.output_markdown else default_md)
    out_json = None if args.no_save_report else (Path(args.output_json) if args.output_json else default_json)

    # Save Markdown output
    if out_md:
        md_lines = [
            f"# LETTER Semantic ID Strategy Comparison Report: {args.dataset}",
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
            rho_val = r["spearman_rho_vs_frequency"]
            rho_str = f"{rho_val:10.4f}" if rho_val is not None else "         -"
            md_lines.append(
                f"| {r['label']:<20} | {r['mean_length']:12.3f} | {r['traffic_weighted_length']:14.3f} | "
                f"{r['token_savings_pct']:11.1f}% | {l1:<5} | {l2:<6} | {l3:<6} | {l4:<6} | {rho_str} |"
            )
        md_lines.append(separator)
        md_lines.append("")

        if pairwise_md_sections:
            md_lines.extend(pairwise_md_sections)

        md_lines.append("## Signal Definitions & Methodologies")
        md_lines.append("")
        for r in results:
            md_lines.append(f"- **{r['label']}** (`{r['signal']}`): {r['basis']}")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with out_md.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved Markdown report to: {out_md}")

    # Save JSON output
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": args.dataset, "results": results}
        if pairwise_match:
            payload["pairwise_exact_match_pct"] = pairwise_match
            payload["pairwise_mae_tokens"] = pairwise_mae
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON comparison data to: {out_json}")


if __name__ == "__main__":
    main()

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
    "co_occurrence": {
        "label": "Co-occurrence Degree",
        "basis": "Unique co-occurring items across user interaction sessions",
        "file_tag": "pop-cooccur",
    },
    "cf_density": {
        "label": "CF Manifold Isolation",
        "basis": "1 - kNN crowd cosine density in CF space",
        "file_tag": "pop-cf",
    },
}


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
        "--strategies",
        type=str,
        default=None,
        help="Optional comma- or space-separated list of strategies (e.g. 'fixed,shortest_unique,popularity:frequency,residual'). If set, overrides --signals and --include-baselines.",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default="frequency,user_entropy,pagerank,co_occurrence,cf_density",
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
        help="Directory to save report files (defaults to <repo-root>/LETTER-TIGER/results/<dataset>).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save comparison JSON (defaults to <output-dir>/semantic_id_comparison.json).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Path to save markdown report (defaults to <output-dir>/semantic_id_comparison.md).",
    )
    parser.add_argument(
        "--no-save-report",
        action="store_true",
        default=False,
        help="Do not save report files to disk; only print to console.",
    )
    return parser.parse_args()


def evaluate_semantic_ids(
    dataset="Instruments",
    tokenizer="rqvae",
    repo_root=None,
    data_root=None,
    index_file=None,
    inter_file=None,
    cf_emb_file=None,
    residuals_file=None,
    residual_threshold=0.2,
    min_length=1,
    max_length=4,
    signals=None,
    include_baselines=True,
    pairwise=True,
    strategies=None,
    existing_sid_entries=None,
    existing_lengths=None,
):
    """Core evaluation function comparing Semantic ID generation strategies on CPU.

    Returns a dictionary containing:
      - 'results': list of per-strategy metric dictionaries
      - 'strategy_lengths': dict mapping strategy -> {item_id: length}
      - 'pairwise_exact_match_pct': dict mapping s1 -> {s2: match_pct}
      - 'pairwise_mae_tokens': dict mapping s1 -> {s2: mae_diff}
    """
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    data_root = Path(data_root) if data_root else repo_root / "data"
    dataset_dir = data_root / dataset
    tok_dir = dataset_dir / tokenizer if tokenizer else dataset_dir

    if index_file is None:
        cand_list = []
        if max_length:
            cand_list.extend([
                tok_dir / f"{dataset}.index.fixed.L{max_length}.json",
                tok_dir / f"{dataset}.index.fixed-for-varlen.L{max_length}.json",
                dataset_dir / f"{dataset}.index.fixed.L{max_length}.json",
                dataset_dir / f"{dataset}.index.fixed-for-varlen.L{max_length}.json",
            ])
        cand_list.append(dataset_dir / f"{dataset}.index.json")
        index_file = next((c for c in cand_list if c.exists()), cand_list[0] if cand_list else tok_dir / f"{dataset}.index.fixed.L4.json")
    index_file = Path(index_file)
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    if inter_file is None:
        inter_file = dataset_dir / f"{dataset}.inter.json"
    inter_file = Path(inter_file)
    if not inter_file.exists():
        raise FileNotFoundError(f"Interaction file not found: {inter_file}")

    if not cf_emb_file:
        cand_cf = repo_root / "RQ-VAE" / "ckpt" / f"{dataset}-32d-sasrec.pt"
        if cand_cf.exists():
            cf_emb_file = str(cand_cf)

    with index_file.open(encoding="utf-8") as f:
        indices = json.load(f)

    if max_length is None and indices:
        first_v = next(iter(indices.values()))
        max_length = len(first_v) if isinstance(first_v, (list, tuple)) else 4

    raw_scores, raw_freqs = compute_interaction_signals(inter_file, signal="frequency")
    items = list(indices.keys())
    total_traffic = max(1, sum(raw_freqs.get(str(i), 0) for i in items))

    results = []
    strategy_lengths = {}

    def add_fixed(depth=None, strat_name=None):
        d = depth if depth is not None else max_length
        s_name = strat_name or ("fixed" if d == max_length else f"fixed_L{d}")
        strategy_lengths[s_name] = {str(i): d for i in items}

        loaded_d_index = None
        d_candidates = [
            tok_dir / f"{dataset}.index.fixed.L{d}.json",
            tok_dir / f"{dataset}.index.fixed-for-varlen.L{d}.json",
            dataset_dir / f"{dataset}.index.fixed.L{d}.json",
            dataset_dir / f"{dataset}.index.fixed-for-varlen.L{d}.json",
        ]
        for dc in d_candidates:
            if dc.exists():
                try:
                    with dc.open(encoding="utf-8") as f:
                        loaded_d_index = json.load(f)
                    break
                except Exception:
                    pass

        if loaded_d_index:
            u_ids = len({tuple(x) for x in loaded_d_index.values()})
            c_count = len(loaded_d_index) - u_ids
        else:
            trunc_fixed = {i: c[:d] for i, c in indices.items()}
            u_ids = len({tuple(c) for c in trunc_fixed.values()})
            c_count = len(trunc_fixed) - u_ids

        results.append({
            "strategy": s_name,
            "phase": "fixed",
            "signal": "fixed",
            "label": f"Fixed (L={d})",
            "basis": f"Uniform fixed codebook depth L={d}",
            "file_tag": s_name,
            "mean_length": float(d),
            "traffic_weighted_length": float(d),
            "token_savings_pct": ((max_length - d) / max_length * 100.0) if max_length > 0 else 0.0,
            "spearman_rho_vs_frequency": None,
            "length_distribution": {d: len(indices)},
            "unique_ids": u_ids,
            "collisions": c_count,
        })

    def add_shortest_unique():
        trunc_su, lens_su = truncate_indices(
            indices,
            min_length=min_length,
            max_length=max_length,
            strategy="shortest_unique",
        )
        strategy_lengths["shortest_unique"] = {str(k): v for k, v in lens_su.items()}
        su_mean = sum(lens_su.values()) / len(lens_su)
        su_traffic_w = sum(lens_su[i] * raw_freqs.get(str(i), 0) for i in trunc_su) / total_traffic
        su_savings = (1.0 - su_traffic_w / max_length) * 100.0
        su_dist = dict(sorted(Counter(lens_su.values()).items()))
        su_uniq = len({tuple(v) for v in trunc_su.values()})
        su_coll = len(trunc_su) - su_uniq
        results.append({
            "strategy": "shortest_unique",
            "phase": "1",
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

    def add_residual():
        if residuals_file:
            res_file_cand = Path(residuals_file)
        elif max_length != 4 and (dataset_dir / f"{dataset}.residuals.L{max_length}.json").exists():
            res_file_cand = dataset_dir / f"{dataset}.residuals.L{max_length}.json"
        else:
            res_file_cand = dataset_dir / f"{dataset}.residuals.json"
        if res_file_cand.exists():
            with res_file_cand.open(encoding="utf-8") as rf:
                loaded_residuals = json.load(rf)
            trunc_res, lens_res = truncate_indices(
                indices,
                min_length=min_length,
                max_length=max_length,
                strategy="residual",
                residuals=loaded_residuals,
                residual_threshold=residual_threshold,
            )
            strategy_lengths["residual"] = {str(k): v for k, v in lens_res.items()}
            res_mean = sum(lens_res.values()) / len(lens_res)
            res_traffic_w = sum(lens_res[i] * raw_freqs.get(str(i), 0) for i in trunc_res) / total_traffic
            res_savings = (1.0 - res_traffic_w / max_length) * 100.0
            res_dist = dict(sorted(Counter(lens_res.values()).items()))
            res_uniq = len({tuple(v) for v in trunc_res.values()})
            res_coll = len(trunc_res) - res_uniq
            results.append({
                "strategy": "residual",
                "phase": "1",
                "signal": "residual",
                "label": f"Residual (Thresh={residual_threshold})",
                "basis": f"Shortest prefix with cumulative reconstruction error <= {residual_threshold}",
                "file_tag": "res",
                "mean_length": res_mean,
                "traffic_weighted_length": res_traffic_w,
                "token_savings_pct": res_savings,
                "spearman_rho_vs_frequency": None,
                "length_distribution": res_dist,
                "unique_ids": res_uniq,
                "collisions": res_coll,
            })
        elif existing_sid_entries and ("residual" in existing_sid_entries or ("residual", "1") in existing_sid_entries):
            prev = dict(existing_sid_entries.get(("residual", "1")) or existing_sid_entries.get("residual"))
            prev["strategy"] = "residual"
            prev["phase"] = "1"
            prev["signal"] = "residual"
            results.append(prev)
            if existing_lengths and "residual" in existing_lengths:
                strategy_lengths["residual"] = existing_lengths["residual"]

    def add_signal(sig, strat_key=None):
        if strat_key is None:
            strat_key = f"popularity:{sig}"
        meta = SIGNAL_METADATA.get(sig, {"label": sig, "basis": "", "file_tag": sig})
        if sig == "cf_density":
            can_compute = bool(cf_emb_file)
            if can_compute:
                try:
                    import torch
                except ImportError:
                    can_compute = False
            if not can_compute:
                prev_cand = None
                if existing_sid_entries:
                    prev_cand = existing_sid_entries.get((strat_key, "1")) or existing_sid_entries.get(strat_key)
                if prev_cand:
                    prev = dict(prev_cand)
                    prev["strategy"] = strat_key
                    prev["phase"] = "1"
                    prev["signal"] = strat_key
                    results.append(prev)
                    if existing_lengths and strat_key in existing_lengths:
                        strategy_lengths[strat_key] = existing_lengths[strat_key]
                return

        scores, _ = compute_interaction_signals(
            inter_file,
            signal=sig,
            cf_emb_file=cf_emb_file if sig == "cf_density" else None,
        )
        truncated, lengths = truncate_indices(
            indices,
            min_length=min_length,
            max_length=max_length,
            strategy="collaborative",
            item_scores=scores,
            item_frequencies=raw_freqs,
        )
        strategy_lengths[strat_key] = {str(k): v for k, v in lengths.items()}

        mean_len = sum(lengths.values()) / len(lengths)
        traffic_w_len = sum(lengths[i] * raw_freqs.get(str(i), 0) for i in truncated) / total_traffic
        token_savings_pct = (1.0 - traffic_w_len / max_length) * 100.0
        length_dist = dict(sorted(Counter(lengths.values()).items()))
        unique_ids = len({tuple(v) for v in truncated.values()})
        collisions = len(truncated) - unique_ids
        rho = compute_spearman_correlation(raw_scores, scores, items)

        results.append({
            "strategy": strat_key,
            "phase": "1",
            "signal": strat_key,
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
        })

    def add_phase1_5(strat, strat_key=None):
        if strat_key is None:
            strat_key = strat
        meta = SIGNAL_METADATA.get(strat, {"label": strat, "basis": "", "file_tag": strat})

        # Determine file suffix
        var_tag = (
            f".max{max_length}" if min_length == 1 else f".min{min_length}-max{max_length}"
        )

        strat_suffix = ""
        sig_name = None
        if strat == "shortest_unique":
            strat_suffix = ""
        elif strat == "residual":
            strat_suffix = ".res"
        elif strat.startswith("popularity") or strat.startswith("collaborative") or strat.startswith("pop-"):
            if ":" in strat:
                sig_name = strat.split(":", 1)[1]
            elif strat.startswith("pop-"):
                sig_name = strat[4:]
            else:
                sig_name = "frequency"
            alias = {
                "pop": "frequency",
                "raw": "frequency",
                "entropy": "user_entropy",
                "pr": "pagerank",
                "cooccur": "co_occurrence",
                "co_occur": "co_occurrence",
                "cooccurrence": "co_occurrence",
                "cf": "cf_density",
            }
            sig_name = alias.get(sig_name, sig_name)
            sig_suffix_map = {
                "frequency": ".pop",
                "user_entropy": ".pop-entropy",
                "pagerank": ".pop-pagerank",
                "co_occurrence": ".pop-cooccur",
                "cf_density": ".pop-cf",
            }
            strat_suffix = sig_suffix_map.get(sig_name, f".pop-{sig_name}")
            meta = SIGNAL_METADATA.get(sig_name, meta)

        cand_files = [
            tok_dir / f"{dataset}.index.varlen{strat_suffix}-phase1.5{var_tag}.json",
            dataset_dir / f"{dataset}.index.varlen{strat_suffix}-phase1.5{var_tag}.json",
        ]
        found_file = next((c for c in cand_files if c.exists()), None)

        if found_file:
            with found_file.open(encoding="utf-8") as f:
                idx_p15 = json.load(f)

            # Check summary file
            sum_file = found_file.with_suffix("").with_suffix(".summary.json")
            if not sum_file.exists():
                sum_file = found_file.parent / (found_file.stem + ".summary.json")

            sum_data = {}
            if sum_file.exists():
                try:
                    with sum_file.open(encoding="utf-8") as sf:
                        sum_data = json.load(sf)
                except Exception:
                    pass

            lens_p15 = {str(k): len(v) for k, v in idx_p15.items()}
            strategy_lengths[(strat_key, "1.5")] = lens_p15

            mean_len = sum_data.get("mean_length") or (sum(lens_p15.values()) / max(1, len(lens_p15)))
            unique_ids = sum_data.get("unique_ids") if "unique_ids" in sum_data else len({tuple(v) for v in idx_p15.values()})
            collisions = sum_data.get("collisions") if "collisions" in sum_data else (len(idx_p15) - unique_ids)
            length_dist = sum_data.get("length_distribution") or dict(sorted(Counter(lens_p15.values()).items()))

            traffic_w_len = sum(len(idx_p15.get(str(i), [])) * raw_freqs.get(str(i), 0) for i in items if str(i) in idx_p15) / total_traffic
            token_savings_pct = (1.0 - traffic_w_len / max_length) * 100.0

            rho = None
            if sig_name and sig_name in SIGNAL_METADATA:
                try:
                    scores, _ = compute_interaction_signals(
                        inter_file,
                        signal=sig_name,
                        cf_emb_file=cf_emb_file if sig_name == "cf_density" else None,
                    )
                    rho = compute_spearman_correlation(scores, lens_p15, items)
                except Exception:
                    pass

            results.append({
                "strategy": strat_key,
                "phase": "1.5",
                "signal": sig_name or strat,
                "label": f"{meta.get('label', strat)} (Phase 1.5)",
                "basis": f"Length-Aware Training (Phase 1.5): {meta.get('basis', '')}".strip(),
                "file_tag": f"{meta.get('file_tag', strat)}-phase1.5",
                "mean_length": mean_len,
                "traffic_weighted_length": traffic_w_len,
                "token_savings_pct": token_savings_pct,
                "spearman_rho_vs_frequency": rho,
                "length_distribution": length_dist,
                "unique_ids": unique_ids,
                "collisions": collisions,
            })
        elif existing_sid_entries and (
            (strat_key, "1.5") in existing_sid_entries
            or f"{strat_key}_phase1.5" in existing_sid_entries
            or f"{strat_key}:phase1.5" in existing_sid_entries
        ):
            prev = dict(
                existing_sid_entries.get((strat_key, "1.5"))
                or existing_sid_entries.get(f"{strat_key}_phase1.5")
                or existing_sid_entries.get(f"{strat_key}:phase1.5")
            )
            prev["strategy"] = strat_key
            prev["phase"] = "1.5"
            results.append(prev)
            if existing_lengths and (strat_key, "1.5") in existing_lengths:
                strategy_lengths[(strat_key, "1.5")] = existing_lengths[(strat_key, "1.5")]
        else:
            results.append({
                "strategy": strat_key,
                "phase": "1.5",
                "signal": sig_name or strat,
                "label": f"{meta.get('label', strat)} (Phase 1.5)",
                "basis": f"Length-Aware Training (Phase 1.5): {meta.get('basis', '')}".strip(),
                "file_tag": f"{meta.get('file_tag', strat)}-phase1.5",
                "mean_length": None,
                "traffic_weighted_length": None,
                "token_savings_pct": None,
                "spearman_rho_vs_frequency": None,
                "length_distribution": {},
                "unique_ids": 0,
                "collisions": 0,
            })

    if strategies is not None:
        for item in strategies:
            if isinstance(item, tuple):
                strat, item_phase = item[0], str(item[1])
            elif isinstance(item, str):
                if item.endswith(":phase1.5") or item.endswith("@phase1.5") or item.endswith("_phase1.5"):
                    item_phase = "1.5"
                    strat = item.rsplit(":", 1)[0].rsplit("@", 1)[0].rsplit("_phase1.5", 1)[0]
                else:
                    strat = item
                    item_phase = "fixed" if (strat == "fixed" or strat.startswith("fixed_L")) else "1"
            else:
                strat = str(item)
                item_phase = "fixed" if (strat == "fixed" or strat.startswith("fixed_L")) else "1"

            if item_phase == "1.5":
                add_phase1_5(strat)
            elif strat == "fixed" or (isinstance(strat, str) and strat.startswith("fixed_L")):
                if strat.startswith("fixed_L") and strat[7:].isdigit():
                    d = int(strat[7:])
                    add_fixed(depth=d, strat_name=strat)
                else:
                    add_fixed()
            elif strat == "shortest_unique":
                add_shortest_unique()
            elif strat == "residual":
                add_residual()
            elif strat.startswith("popularity") or strat.startswith("collaborative") or strat.startswith("pop-"):
                if ":" in strat:
                    sig = strat.split(":", 1)[1]
                elif strat.startswith("pop-"):
                    sig = strat[4:]
                else:
                    sig = "frequency"
                alias = {
                    "pop": "frequency",
                    "raw": "frequency",
                    "entropy": "user_entropy",
                    "pr": "pagerank",
                    "cooccur": "co_occurrence",
                    "co_occur": "co_occurrence",
                    "cooccurrence": "co_occurrence",
                    "cf": "cf_density",
                }
                sig = alias.get(sig, sig)
                if sig in SIGNAL_METADATA:
                    add_signal(sig, strat_key=strat)
    else:
        if include_baselines:
            add_fixed()
            add_shortest_unique()
            add_residual()

        if signals is None:
            sig_list = ["frequency", "user_entropy", "pagerank", "co_occurrence", "cf_density"]
        elif isinstance(signals, str):
            sig_list = [s.strip() for s in signals.replace(" ", ",").split(",") if s.strip()]
        else:
            sig_list = list(signals)

        for sig in sig_list:
            if sig in SIGNAL_METADATA:
                add_signal(sig, strat_key=sig)

    pairwise_match = {}
    pairwise_mae = {}
    has_phase15 = any(r.get("phase") == "1.5" for r in results)

    pairwise_items = []
    for r in results:
        strat = r["strategy"]
        ph = str(r.get("phase", "1"))
        l_key = (strat, ph) if (strat, ph) in strategy_lengths else (strat if strat in strategy_lengths else None)
        if l_key and l_key in strategy_lengths:
            if has_phase15 and ph not in ("fixed", "-"):
                display_label = f"{strat} (P{ph})"
            else:
                display_label = strat
            pairwise_items.append((display_label, l_key))

    if pairwise and len(pairwise_items) > 1:
        n_items = len(items)
        for d1, k1 in pairwise_items:
            pairwise_match[d1] = {}
            pairwise_mae[d1] = {}
            l1 = strategy_lengths[k1]
            for d2, k2 in pairwise_items:
                l2 = strategy_lengths[k2]
                exact_matches = sum(l1.get(str(i), 0) == l2.get(str(i), 0) for i in items)
                pairwise_match[d1][d2] = round((exact_matches / n_items) * 100.0, 2)
                pairwise_mae[d1][d2] = round(
                    sum(abs(l1.get(str(i), 0) - l2.get(str(i), 0)) for i in items) / n_items, 3
                )

    return {
        "dataset": dataset,
        "items_count": len(indices),
        "total_traffic": total_traffic,
        "min_length": min_length,
        "max_length": max_length,
        "results": results,
        "strategy_lengths": strategy_lengths,
        "pairwise_exact_match_pct": pairwise_match,
        "pairwise_mae_tokens": pairwise_mae,
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root) if args.data_root else repo_root / "data"

    parsed_strategies = None
    if args.strategies:
        raw_s = args.strategies.replace(",", " ").split()
        parsed_strategies = [s.strip() for s in raw_s if s.strip()]

    eval_data = evaluate_semantic_ids(
        dataset=args.dataset,
        repo_root=repo_root,
        data_root=data_root,
        index_file=args.index_file,
        inter_file=args.inter_file,
        cf_emb_file=args.cf_emb_file,
        residuals_file=args.residuals_file,
        residual_threshold=args.residual_threshold,
        min_length=args.min_length,
        max_length=args.max_length,
        signals=args.signals,
        include_baselines=args.include_baselines,
        pairwise=args.pairwise,
        strategies=parsed_strategies,
    )

    results = eval_data["results"]
    pairwise_match = eval_data["pairwise_exact_match_pct"]
    pairwise_mae = eval_data["pairwise_mae_tokens"]

    # Table output
    layer_keys = list(range(1, args.max_length + 1))
    layer_headers = [f"L={k}" for k in layer_keys]
    layer_widths = [max(6, len(h)) for h in layer_headers]
    strat_col_w = max(20, max(len(r["label"]) for r in results) if results else 20)

    layer_hdr_str = " | ".join(f"{h:>{w}}" for h, w in zip(layer_headers, layer_widths))
    layer_sep_str = "|".join("-" * (w + 2) for w in layer_widths)
    header = (
        f"| {'Signal':<{strat_col_w}} | {'Catalog Mean':>12} | {'Traffic W-Len':>14} | {'Token Savings':>13} | "
        f"{layer_hdr_str} | {'Collisions':<10} | {'Spearman ρ':>10} |"
    )
    separator = (
        f"|{'-'*(strat_col_w + 2)}|{'-'*14}|{'-'*16}|{'-'*15}|"
        f"{layer_sep_str}|{'-'*12}|{'-'*12}|"
    )

    tot_items = eval_data.get("items_count", 0)
    print("\n" + "=" * len(separator))
    print(f" LETTER Semantic ID Strategy Comparison for Dataset: {args.dataset}")
    print(f" Catalog Items: {tot_items:,} | Total Interactions: {eval_data['total_traffic']:,}")
    print(f" Length Range: [{args.min_length}, {args.max_length}]")
    print("=" * len(separator))
    print(header)
    print(separator)

    for r in results:
        d = r.get("length_distribution", {})
        tot = tot_items or (sum(d.values()) if d else 0)
        if tot and tot > 0 and d and any(d.values()):
            layer_vals = [
                f"{(d.get(k, d.get(str(k), 0)) / tot) * 100.0:.1f}%"
                for k in layer_keys
            ]
        else:
            layer_vals = ["-" for _ in layer_keys]
        layer_val_str = " | ".join(f"{v:>{w}}" for v, w in zip(layer_vals, layer_widths))
        rho_val = r.get("spearman_rho_vs_frequency")
        rho_str = f"{rho_val:10.4f}" if rho_val is not None else "         -"
        coll_str = f"{r.get('collisions', 0):<10}"
        print(
            f"| {r['label']:<{strat_col_w}} | {r['mean_length']:12.3f} | {r['traffic_weighted_length']:14.3f} | "
            f"{r['token_savings_pct']:11.1f}% | {layer_val_str} | {coll_str} | {rho_str} |"
        )
    print(separator)

    # Pairwise printing
    if pairwise_match:
        strat_keys = [r["signal"] for r in results]
        strat_labels = [r["label"] for r in results]
        col_w = max(14, max(len(l) for l in strat_labels) + 2)
        row_fmt = f"| {{:<{col_w}}} | " + " | ".join([f"{{:<{col_w}}}" for _ in strat_labels]) + " |"
        sep_fmt = f"+{'-' * (col_w + 2)}+" + "+".join([f"{'-' * (col_w + 2)}" for _ in strat_labels]) + "+"

        print("\n" + "=" * len(sep_fmt))
        print(" Pairwise Exact Semantic ID Agreement Rate (%)")
        print("=" * len(sep_fmt))
        print(row_fmt.format("Strategy", *strat_labels))
        print(sep_fmt)
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in strat_keys]
            print(row_fmt.format(l1, *vals))
        print(sep_fmt)

        print("\n" + "=" * len(sep_fmt))
        print(" Pairwise Mean Absolute Length Difference (Tokens)")
        print("=" * len(sep_fmt))
        print(row_fmt.format("Strategy", *strat_labels))
        print(sep_fmt)
        for s1, l1 in zip(strat_keys, strat_labels):
            vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in strat_keys]
            print(row_fmt.format(l1, *vals))
        print(sep_fmt)

    # Output paths: default to LETTER-TIGER/results/<dataset>
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = repo_root / "LETTER-TIGER" / "results" / args.dataset

    rep_tag = f"_max{args.max_length}" if args.min_length == 1 else f"_min{args.min_length}-max{args.max_length}"
    default_md = out_dir / f"semantic_id_comparison{rep_tag}.md"
    default_json = out_dir / f"semantic_id_comparison{rep_tag}.json"

    out_md = None if args.no_save_report else (Path(args.output_markdown) if args.output_markdown else default_md)
    out_json = None if args.no_save_report else (Path(args.output_json) if args.output_json else default_json)

    if out_md:
        md_lines = [
            f"# LETTER Semantic ID Strategy Comparison Report: {args.dataset}",
            "",
            f"- **Dataset**: `{args.dataset}`",
            f"- **Items**: {eval_data['items_count']:,}",
            f"- **Fixed ID Depth**: {args.max_length} tokens",
            f"- **Minimum Allowable Depth**: {args.min_length} token",
            "",
            "## 1. Semantic ID Evaluation & Compression",
            "",
            (
                "| Strategy | Catalog Mean | Traffic W-Len | Token Savings | "
                + " | ".join(layer_headers)
                + " | Collisions | Spearman ρ |"
            ),
            (
                "| :--- | :---: | :---: | :---: | "
                + " | ".join([":---:" for _ in layer_keys])
                + " | :---: | :---: |"
            ),
        ]
        for r in results:
            d = r.get("length_distribution", {})
            tot = eval_data.get("items_count", 0) or (sum(d.values()) if d else 0)
            if tot and tot > 0 and d and any(d.values()):
                layer_vals = [
                    f"{(d.get(k, d.get(str(k), 0)) / tot) * 100.0:.1f}%"
                    for k in layer_keys
                ]
            else:
                layer_vals = ["-" for _ in layer_keys]
            rho_val = r.get("spearman_rho_vs_frequency")
            rho_str = f"{rho_val:.4f}" if rho_val is not None else "-"
            md_lines.append(
                f"| **{r['label']}** | {r['mean_length']:.3f} | {r['traffic_weighted_length']:.3f} | "
                f"{r['token_savings_pct']:.1f}% | "
                + " | ".join(layer_vals)
                + f" | {r.get('collisions', 0)} | {rho_str} |"
            )
        md_lines.append("")

        if pairwise_match:
            strat_keys = [r["signal"] for r in results]
            strat_labels = [r["label"] for r in results]
            md_header = "| Strategy | " + " | ".join([f"{l}" for l in strat_labels]) + " |"
            md_sep = "| :--- | " + " | ".join([":---:" for _ in strat_labels]) + " |"

            md_lines.extend([
                "## 2. Pairwise Exact Semantic ID Agreement Rate (%)",
                "",
                "> Percentage of items in catalog that receive an identical Semantic ID across strategies.",
                "",
                md_header,
                md_sep,
            ])
            for s1, l1 in zip(strat_keys, strat_labels):
                row_vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in strat_keys]
                md_lines.append(f"| **{l1}** | " + " | ".join(row_vals) + " |")
            md_lines.append("")

            md_lines.extend([
                "## 3. Pairwise Mean Absolute Length Difference (Tokens)",
                "",
                "> Average token length difference per item (|L_A - L_B|). Lower value indicates closer length profiles.",
                "",
                md_header,
                md_sep,
            ])
            for s1, l1 in zip(strat_keys, strat_labels):
                row_vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in strat_keys]
                md_lines.append(f"| **{l1}** | " + " | ".join(row_vals) + " |")
            md_lines.append("")

        out_md.parent.mkdir(parents=True, exist_ok=True)
        with out_md.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved Markdown report to: {out_md}")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "semantic_id_metrics": results,
            "pairwise_exact_match_pct": pairwise_match,
            "pairwise_mae_tokens": pairwise_mae,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON comparison data to: {out_json}")


if __name__ == "__main__":
    main()

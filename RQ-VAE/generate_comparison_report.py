import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Add RQ-VAE directory to sys.path for local module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from truncate_indices import (
        truncate_indices,
        compute_interaction_signals,
    )
except ImportError:
    truncate_indices = None
    compute_interaction_signals = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison report and tables across LETTER strategies."
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
    parser.add_argument("--max-length", type=int, default=4, help="Maximum SID length.")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum SID length.")
    parser.add_argument(
        "--strategies",
        type=str,
        default="fixed,shortest_unique,popularity,residual",
        help="Comma- or space-separated list of strategies.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tiger",
        help="Comma-separated list of models (e.g. tiger,lcrec).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge with existing strategy_comparison.json or auto-discover existing result files.",
    )
    parser.add_argument(
        "--no-pairwise-sid",
        action="store_true",
        help="Do not compute pairwise Semantic ID agreement and length divergence tables.",
    )
    return parser.parse_args()


def file_to_strategy(filename, tag=""):
    """Infer strategy name from result JSON filename."""
    if not filename.endswith(".json") or filename.startswith("strategy_comparison"):
        return None
    stem = filename[:-5]
    if tag and stem.endswith(tag):
        stem = stem[:-len(tag)]
    if stem == "fixed" or stem.startswith("fixed_L"):
        return "fixed"
    if stem == "varlen":
        return "shortest_unique"
    if stem == "varlen-res":
        return "residual"
    if stem == "varlen-pop":
        return "popularity:frequency"
    if stem.startswith("varlen-pop-"):
        sig = stem[len("varlen-pop-"):]
        alias = {"entropy": "user_entropy", "pr": "pagerank", "cf": "cf_density"}
        return f"popularity:{alias.get(sig, sig)}"
    return None


def strategy_sort_key(strat_name):
    """Sort order: fixed -> shortest_unique -> popularity signals -> residual -> others."""
    if strat_name == "fixed":
        return (0, 0, strat_name)
    if strat_name == "shortest_unique":
        return (1, 0, strat_name)
    if strat_name.startswith("popularity") or strat_name.startswith("collaborative"):
        sig = strat_name.split(":", 1)[1] if ":" in strat_name else "frequency"
        collab_order = {
            "frequency": 1,
            "raw": 1,
            "pop": 1,
            "user_entropy": 2,
            "entropy": 2,
            "pagerank": 3,
            "pr": 3,
            "target": 4,
            "target_frequency": 4,
            "composite": 5,
            "cf_density": 6,
            "cf_isolation": 6,
        }
        return (2, collab_order.get(sig, 50), strat_name)
    if strat_name == "residual":
        return (3, 0, strat_name)
    return (4, 0, strat_name)


def main():
    args = parse_args()
    dataset = args.dataset
    repo_root = args.repo_root
    data_root = args.data_root or os.path.join(repo_root, "data")
    max_length = args.max_length
    min_length = args.min_length

    # Parse strategies and models
    raw_strats = args.strategies.replace(",", " ").split()
    strategies = [s.strip() for s in raw_strats if s.strip()]

    raw_models = args.models.replace(",", " ").split()
    models = [m.strip().lower() for m in raw_models if m.strip()]

    tag = "" if (max_length == 4 and min_length == 1) else f"_max{max_length}"

    for model in models:
        model_name = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        report_dir = os.path.join(repo_root, model_dir, "results", dataset)
        os.makedirs(report_dir, exist_ok=True)
        report_json = os.path.join(report_dir, "strategy_comparison.json")

        existing_entries = {}
        if not args.no_merge and os.path.isfile(report_json):
            try:
                with open(report_json) as f:
                    prev_raw = json.load(f)
                prev_list = prev_raw.get("strategy_comparison", prev_raw) if isinstance(prev_raw, dict) else prev_raw
                if isinstance(prev_list, list):
                    for item in prev_list:
                        if isinstance(item, dict) and "strategy" in item:
                            s_name = item["strategy"]
                            if s_name == "popularity":
                                s_name = "popularity:frequency"
                            existing_entries[s_name] = item
            except Exception:
                pass

        discovered_strategies = []
        if not args.no_merge and os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                s = file_to_strategy(fname, tag)
                if s and s not in discovered_strategies:
                    discovered_strategies.append(s)

        # Build comprehensive list of strategies:
        # 1) Start with user-requested strategies (normalizing 'popularity' -> 'popularity:frequency' if collab signals used)
        has_collab_expanded = any(
            k.startswith("popularity:") and k != "popularity:frequency"
            for k in list(existing_entries.keys()) + discovered_strategies
        )
        combined_strategies = []
        for s in strategies:
            norm_s = "popularity:frequency" if (s == "popularity" and has_collab_expanded) else s
            if norm_s not in combined_strategies:
                combined_strategies.append(norm_s)

        # 2) Include discovered result files
        for s in discovered_strategies:
            if s not in combined_strategies:
                combined_strategies.append(s)

        # 3) Include previous report entries
        for s in existing_entries:
            if s not in combined_strategies:
                combined_strategies.append(s)

        # Sort canonically
        combined_strategies.sort(key=strategy_sort_key)

        table_data = []
        strategy_lengths = {}
        base_idx_file = os.path.join(data_root, dataset, f"{dataset}.index.json")
        if not os.path.isfile(base_idx_file):
            base_idx_file = os.path.join(data_root, dataset, f"{dataset}.index.fixed.json")

        base_indices = None
        if os.path.isfile(base_idx_file):
            try:
                with open(base_idx_file, encoding="utf-8") as bf:
                    base_indices = json.load(bf)
            except Exception:
                pass

        inter_file = os.path.join(data_root, dataset, f"{dataset}.inter.json")
        residuals_file = os.path.join(data_root, dataset, f"{dataset}.residuals.json")

        for strat in combined_strategies:
            idx_summary_file = None
            res_file = None
            file_tag = "pop"

            if strat == "fixed":
                idx_summary_file = os.path.join(
                    data_root, dataset, f"{dataset}.index.fixed-for-varlen.summary.json"
                )
                if not os.path.isfile(idx_summary_file):
                    idx_summary_file = os.path.join(
                        data_root, dataset, f"{dataset}.index.fixed.summary.json"
                    )
                res_fname = "fixed.json" if max_length == 4 else f"fixed_L{max_length}.json"
                res_file = os.path.join(repo_root, model_dir, "results", dataset, res_fname)

            elif strat == "shortest_unique":
                idx_summary_file = os.path.join(
                    data_root, dataset, f"{dataset}.index.varlen{tag}.summary.json"
                )
                res_file = os.path.join(repo_root, model_dir, "results", dataset, f"varlen{tag}.json")

            elif (
                strat == "popularity"
                or strat.startswith("popularity:")
                or strat.startswith("collaborative:")
                or strat.startswith("pop-")
            ):
                if ":" in strat:
                    sig = strat.split(":", 1)[1]
                elif strat.startswith("pop-"):
                    sig = strat[4:]
                else:
                    sig = "frequency"

                sig_map = {
                    "frequency": ("pop", "-pop"),
                    "raw": ("pop", "-pop"),
                    "user_entropy": ("pop-entropy", "-pop-entropy"),
                    "entropy": ("pop-entropy", "-pop-entropy"),
                    "pagerank": ("pop-pagerank", "-pop-pagerank"),
                    "pr": ("pop-pagerank", "-pop-pagerank"),
                    "target": ("pop-target", "-pop-target"),
                    "target_frequency": ("pop-target", "-pop-target"),
                    "composite": ("pop-composite", "-pop-composite"),
                    "cf_density": ("pop-cf", "-pop-cf"),
                }
                file_tag, res_tag = sig_map.get(sig, (f"pop-{sig}", f"-pop-{sig}"))
                idx_summary_file = os.path.join(
                    data_root, dataset, f"{dataset}.index.varlen.{file_tag}{tag}.summary.json"
                )
                if not os.path.isfile(idx_summary_file):
                    idx_summary_file = os.path.join(
                        data_root, dataset, f"{dataset}.index.varlen.{file_tag}.summary.json"
                    )
                full_res_tag = res_tag if not tag else f"{res_tag}{tag}"
                res_file = os.path.join(
                    repo_root, model_dir, "results", dataset, f"varlen{full_res_tag}.json"
                )

            elif strat == "residual":
                idx_summary_file = os.path.join(
                    data_root, dataset, f"{dataset}.index.varlen.res{tag}.summary.json"
                )
                res_tag = "-res" if not tag else f"-res{tag}"
                res_file = os.path.join(
                    repo_root, model_dir, "results", dataset, f"varlen{res_tag}.json"
                )

            mean_l = float(max_length) if strat == "fixed" else None
            w_mean_l = float(max_length) if strat == "fixed" else None
            collisions = 0

            # 1) Try reading from summary JSON file
            if idx_summary_file and os.path.isfile(idx_summary_file):
                try:
                    with open(idx_summary_file) as f:
                        sdata = json.load(f)
                    mean_l = sdata.get("mean_length", mean_l)
                    w_mean_l = sdata.get("weighted_mean_length", w_mean_l)
                    collisions = sdata.get("collisions", 0)
                except Exception:
                    pass
            elif strat != "fixed":
                # Fallback: inspect actual index file directly
                if strat == "shortest_unique":
                    actual_idx_name = f"{dataset}.index.varlen{tag}.json"
                elif strat == "residual":
                    actual_idx_name = f"{dataset}.index.varlen.res{tag}.json"
                else:
                    actual_idx_name = f"{dataset}.index.varlen.{file_tag}{tag}.json"
                actual_idx_file = os.path.join(data_root, dataset, actual_idx_name)
                if os.path.isfile(actual_idx_file):
                    try:
                        with open(actual_idx_file) as f:
                            idata = json.load(f)
                        lens = [len(v) for v in idata.values()]
                        mean_l = sum(lens) / max(1, len(lens))
                        uniq = len({tuple(v) for v in idata.values()})
                        collisions = len(idata) - uniq
                        strategy_lengths[strat] = {str(k): len(v) for k, v in idata.items()}
                    except Exception:
                        pass

            # Resolve item lengths for pairwise analysis
            if strat == "fixed" and base_indices:
                strategy_lengths["fixed"] = {str(k): max_length for k in base_indices}
            elif strat != "fixed" and strat not in strategy_lengths and base_indices and truncate_indices:
                # Derive lengths dynamically using truncate_indices if index JSON wasn't on disk
                if strat == "shortest_unique":
                    try:
                        _, lmap = truncate_indices(base_indices, min_length=min_length, max_length=max_length, strategy="shortest_unique")
                        strategy_lengths["shortest_unique"] = {str(k): v for k, v in lmap.items()}
                    except Exception:
                        pass
                elif strat.startswith("popularity") or strat.startswith("collaborative") or strat.startswith("pop-"):
                    if ":" in strat:
                        sig_name = strat.split(":", 1)[1]
                    elif strat.startswith("pop-"):
                        sig_name = strat[4:]
                    else:
                        sig_name = "frequency"
                    alias = {"entropy": "user_entropy", "pr": "pagerank", "cf": "cf_density"}
                    sig_name = alias.get(sig_name, sig_name)
                    if os.path.isfile(inter_file) and compute_interaction_signals:
                        try:
                            scores, freqs = compute_interaction_signals(inter_file, signal=sig_name)
                            _, lmap = truncate_indices(base_indices, min_length=min_length, max_length=max_length, strategy="popularity", item_scores=scores, item_frequencies=freqs)
                            strategy_lengths[strat] = {str(k): v for k, v in lmap.items()}
                        except Exception:
                            pass
                elif strat == "residual":
                    if os.path.isfile(residuals_file):
                        try:
                            with open(residuals_file) as rf:
                                rdata = json.load(rf)
                            _, lmap = truncate_indices(base_indices, min_length=min_length, max_length=max_length, strategy="residual", residuals=rdata, residual_threshold=0.2)
                            strategy_lengths["residual"] = {str(k): v for k, v in lmap.items()}
                        except Exception:
                            pass

                if mean_l is None and strat in strategy_lengths:
                    lens = list(strategy_lengths[strat].values())
                    if lens:
                        mean_l = sum(lens) / len(lens)

            # 2) Fallback for weighted mean length
            if w_mean_l is None and strat != "fixed":
                if strat in strategy_lengths and os.path.isfile(inter_file):
                    try:
                        with open(inter_file) as int_f:
                            idata = json.load(int_f)
                        freqs = Counter()
                        for v in idata.values():
                            if isinstance(v, list):
                                freqs.update(str(x) for x in v)
                            elif isinstance(v, (int, float)):
                                freqs[str(v)] += 1
                        tot_f = sum(freqs.get(str(i), 0) for i in strategy_lengths[strat])
                        if tot_f > 0:
                            w_mean_l = sum(
                                l * freqs.get(str(i), 0) for i, l in strategy_lengths[strat].items()
                            ) / tot_f
                    except Exception:
                        pass
                else:
                    if strat == "shortest_unique":
                        actual_idx_name = f"{dataset}.index.varlen{tag}.json"
                    elif strat == "residual":
                        actual_idx_name = f"{dataset}.index.varlen.res{tag}.json"
                    else:
                        actual_idx_name = f"{dataset}.index.varlen.{file_tag}{tag}.json"
                    idx_path = os.path.join(data_root, dataset, actual_idx_name)
                    inter_path = os.path.join(data_root, dataset, f"{dataset}.inter.json")
                    if os.path.isfile(idx_path) and os.path.isfile(inter_path):
                        try:
                            with open(inter_path) as int_f:
                                idata = json.load(int_f)
                            freqs = Counter()
                            for v in idata.values():
                                if isinstance(v, list):
                                    freqs.update(str(x) for x in v)
                                elif isinstance(v, (int, float)):
                                    freqs[str(v)] += 1
                            with open(idx_path) as ifile:
                                idx_data = json.load(ifile)
                            tot_f = sum(freqs.get(str(i), 0) for i in idx_data)
                            if tot_f > 0:
                                w_mean_l = sum(
                                    len(toks) * freqs.get(str(i), 0) for i, toks in idx_data.items()
                                ) / tot_f
                        except Exception:
                            pass

            metrics = {}
            status = "pending"
            if res_file and os.path.isfile(res_file):
                try:
                    with open(res_file) as f:
                        rdata = json.load(f)
                    metrics = rdata.get("mean_results", {})
                    status = "completed"
                except Exception:
                    status = "error"

            # 3) Fallback / merge from existing entries in strategy_comparison.json
            if strat in existing_entries:
                prev_row = existing_entries[strat]
                if status != "completed" and prev_row.get("status") == "completed":
                    metrics = prev_row.get("metrics", {})
                    status = "completed"
                if mean_l is None and prev_row.get("mean_length") is not None:
                    mean_l = prev_row["mean_length"]
                if w_mean_l is None and prev_row.get("weighted_length") is not None:
                    w_mean_l = prev_row["weighted_length"]
                if collisions == 0 and prev_row.get("collisions", 0) > 0:
                    collisions = prev_row["collisions"]

            table_data.append({
                "strategy": strat,
                "mean_length": mean_l,
                "weighted_length": w_mean_l,
                "collisions": collisions,
                "metrics": metrics,
                "status": status,
            })

        # Locate fixed baseline metrics if present
        fixed_metrics = None
        for row in table_data:
            if row["strategy"] == "fixed" and row["status"] == "completed":
                fixed_metrics = row["metrics"]
                break

        # Dynamically size the strategy column for alignment
        max_s_len = max([len(r["strategy"]) for r in table_data] + [len("Strategy")])
        strat_col_w = max(22, max_s_len)
        row_fmt = f"| {{:<{strat_col_w}}} | {{:<8}} | {{:<10}} | {{:<6}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} |"
        sep = f"+{'-' * (strat_col_w + 2)}+----------+------------+--------+----------+----------+----------+----------+----------+"
        header = row_fmt.format(
            "Strategy", "Mean L", "W-Mean L", "Coll.", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10"
        )
        table_width = len(sep)

        lines = [
            "\n" + "=" * table_width,
            f" STRATEGY COMPARISON EXPERIMENT: {model_name} ({dataset})",
            "=" * table_width,
            header,
            sep,
        ]

        for row in table_data:
            strat = row["strategy"]
            ml_val = row["mean_length"]
            wml_val = row["weighted_length"]
            ml_str = f"{ml_val:.2f}" if ml_val is not None else "-"
            wml_str = f"{wml_val:.2f}" if wml_val is not None else "-"
            col_str = str(row["collisions"])
            m = row["metrics"]

            if row["status"] == "completed":
                h1 = f"{m.get('hit@1', 0)*100:.2f}%"
                h5 = f"{m.get('hit@5', 0)*100:.2f}%"
                h10 = f"{m.get('hit@10', 0)*100:.2f}%"
                n5 = f"{m.get('ndcg@5', 0)*100:.2f}%"
                n10 = f"{m.get('ndcg@10', 0)*100:.2f}%"
                lines.append(row_fmt.format(strat, ml_str, wml_str, col_str, h1, h5, h10, n5, n10))

                # Add Delta row if not fixed and fixed exists
                if strat != "fixed" and fixed_metrics:
                    def get_delta(key):
                        base = fixed_metrics.get(key, 0)
                        val = m.get(key, 0)
                        if base > 0:
                            diff = (val - base) / base * 100
                            return f"{diff:+.1f}%"
                        return "-"

                    d_ml = f"{((ml_val - max_length) / max_length * 100):+.1f}%" if ml_val is not None else ""
                    d_wml = f"{((wml_val - max_length) / max_length * 100):+.1f}%" if wml_val is not None else ""
                    dh1 = get_delta("hit@1")
                    dh5 = get_delta("hit@5")
                    dh10 = get_delta("hit@10")
                    dn5 = get_delta("ndcg@5")
                    dn10 = get_delta("ndcg@10")
                    delta_line = row_fmt.format("  (delta vs fix)", d_ml, d_wml, "", dh1, dh5, dh10, dn5, dn10)
                    lines.append(delta_line)
            else:
                status_tag = f"({row['status']})"
                lines.append(row_fmt.format(strat, ml_str, wml_str, col_str, status_tag, "-", "-", "-", "-"))

        lines.append(sep)

        pairwise_match = {}
        pairwise_mae = {}
        pairwise_text_sections = []
        pairwise_md_sections = []

        if not args.no_pairwise_sid and len(strategy_lengths) > 1:
            valid_strats = [s for s in combined_strategies if s in strategy_lengths]
            if len(valid_strats) > 1:
                # Find common items across all strategies with lengths
                common_items = sorted(
                    set.intersection(*[set(strategy_lengths[s].keys()) for s in valid_strats]),
                    key=lambda x: int(x) if x.isdigit() else x,
                )
                if common_items:
                    n_items = len(common_items)
                    for s1 in valid_strats:
                        pairwise_match[s1] = {}
                        pairwise_mae[s1] = {}
                        l1 = strategy_lengths[s1]
                        for s2 in valid_strats:
                            l2 = strategy_lengths[s2]
                            matches = sum(l1[k] == l2[k] for k in common_items)
                            diffs = sum(abs(l1[k] - l2[k]) for k in common_items)
                            pairwise_match[s1][s2] = round((matches / n_items) * 100.0, 2)
                            pairwise_mae[s1][s2] = round(diffs / n_items, 3)

                    # Text Formatting
                    max_name_w = max(len(s) for s in valid_strats + ["Strategy"])
                    col_w = max(14, max_name_w + 2)
                    p_row_fmt = f"| {{:<{col_w}}} | " + " | ".join([f"{{:<{col_w}}}" for _ in valid_strats]) + " |"
                    p_sep = f"+{'-' * (col_w + 2)}+" + "+".join([f"{'-' * (col_w + 2)}" for _ in valid_strats]) + "+"
                    p_width = len(p_sep)

                    pairwise_text_sections.extend([
                        "\n" + "=" * p_width,
                        " Pairwise Exact Semantic ID Agreement Rate (%)",
                        " (Percentage of items in catalog assigned identical Semantic IDs between strategies)",
                        "=" * p_width,
                        p_row_fmt.format("Strategy", *valid_strats),
                        p_sep,
                    ])
                    for s1 in valid_strats:
                        row_vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in valid_strats]
                        pairwise_text_sections.append(p_row_fmt.format(s1, *row_vals))
                    pairwise_text_sections.append(p_sep)

                    pairwise_text_sections.extend([
                        "\n" + "=" * p_width,
                        " Pairwise Mean Absolute Length Difference (Tokens)",
                        " (Average token length divergence per item: sum(|L_A - L_B|) / N)",
                        "=" * p_width,
                        p_row_fmt.format("Strategy", *valid_strats),
                        p_sep,
                    ])
                    for s1 in valid_strats:
                        row_vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in valid_strats]
                        pairwise_text_sections.append(p_row_fmt.format(s1, *row_vals))
                    pairwise_text_sections.append(p_sep)

                    # Markdown Formatting
                    md_header = "| Strategy | " + " | ".join([f"{s}" for s in valid_strats]) + " |"
                    md_sep = "| :--- | " + " | ".join([":---:" for _ in valid_strats]) + " |"

                    pairwise_md_sections.extend([
                        "",
                        "### Pairwise Exact Semantic ID Agreement Rate (%)",
                        "",
                        "> Percentage of items in catalog that receive an identical Semantic ID across strategies.",
                        "",
                        md_header,
                        md_sep,
                    ])
                    for s1 in valid_strats:
                        row_vals = [f"{pairwise_match[s1][s2]:.2f}%" for s2 in valid_strats]
                        pairwise_md_sections.append(f"| **{s1}** | " + " | ".join(row_vals) + " |")

                    pairwise_md_sections.extend([
                        "",
                        "### Pairwise Mean Absolute Length Difference (Tokens)",
                        "",
                        "> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.",
                        "",
                        md_header,
                        md_sep,
                    ])
                    for s1 in valid_strats:
                        row_vals = [f"{pairwise_mae[s1][s2]:.3f}" for s2 in valid_strats]
                        pairwise_md_sections.append(f"| **{s1}** | " + " | ".join(row_vals) + " |")

        if pairwise_text_sections:
            lines.extend(pairwise_text_sections)

        summary_text = "\n".join(lines)
        print(summary_text)

        # Save reports
        report_txt = os.path.join(report_dir, "strategy_comparison.txt")
        with open(report_txt, "w") as f:
            f.write(summary_text + "\n")

        report_payload = {
            "strategy_comparison": table_data,
        }
        if pairwise_match:
            report_payload["pairwise_exact_match_pct"] = pairwise_match
            report_payload["pairwise_mae_tokens"] = pairwise_mae
        with open(report_json, "w") as f:
            json.dump(report_payload, f, indent=2)

        # Save markdown version
        md_lines = [
            f"### LETTER Strategy Comparison: {model_name} ({dataset})\n",
            "| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for row in table_data:
            m = row["metrics"]
            ml_val = row["mean_length"]
            wml_val = row["weighted_length"]
            ml = f"{ml_val:.2f}" if ml_val is not None else "-"
            wml = f"{wml_val:.2f}" if wml_val is not None else "-"
            strat_name = row["strategy"]
            st = row["status"]
            col = row["collisions"]
            if st == "completed":
                h1 = f"{m.get('hit@1', 0)*100:.2f}%"
                h5 = f"{m.get('hit@5', 0)*100:.2f}%"
                h10 = f"{m.get('hit@10', 0)*100:.2f}%"
                n5 = f"{m.get('ndcg@5', 0)*100:.2f}%"
                n10 = f"{m.get('ndcg@10', 0)*100:.2f}%"
                md_lines.append(f"| **{strat_name}** | {ml} | {wml} | {col} | {h1} | {h5} | {h10} | {n5} | {n10} | {st} |")
            else:
                md_lines.append(f"| **{strat_name}** | {ml} | {wml} | {col} | - | - | - | - | - | {st} |")

        if pairwise_md_sections:
            md_lines.extend(pairwise_md_sections)

        report_md = os.path.join(report_dir, "strategy_comparison.md")
        with open(report_md, "w") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"Saved text report to:     {report_txt}")
        print(f"Saved JSON report to:     {report_json}")
        print(f"Saved Markdown report to: {report_md}\n")


if __name__ == "__main__":
    main()

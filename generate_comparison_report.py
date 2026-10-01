import argparse
import json
import os
import sys
from pathlib import Path

# Add RQ-VAE directory to sys.path for local module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent / "RQ-VAE"))
from compare_semantic_ids import evaluate_semantic_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison report and tables across LETTER strategies."
    )
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
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
        "--tokenizer",
        type=str,
        default="rqvae",
        help="Tokenizer type: rqvae or letter (default: rqvae).",
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
    if tag:
        if not stem.endswith(tag):
            expected_fixed = f"fixed_L{tag.lstrip('_max')}" if tag.startswith("_max") else None
            if expected_fixed and stem == expected_fixed:
                return "fixed"
            return None
        stem = stem[:-len(tag)]
    else:
        if "_max" in stem or "_min" in stem or (stem.startswith("fixed_L") and stem != "fixed_L4"):
            return None
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
        alias = {
            "entropy": "user_entropy",
            "pr": "pagerank",
            "cooccur": "co_occurrence",
            "co_occur": "co_occurrence",
            "cooccurrence": "co_occurrence",
            "cf": "cf_density",
        }
        norm_sig = alias.get(sig, sig)
        valid_sigs = {"frequency", "user_entropy", "pagerank", "co_occurrence", "cf_density"}
        if norm_sig in valid_sigs:
            return f"popularity:{norm_sig}"
        return None
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
            "co_occurrence": 4,
            "cooccurrence": 4,
            "cooccur": 4,
            "cf_density": 5,
            "cf_isolation": 5,
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

    tokenizer = (args.tokenizer or "rqvae").strip().lower()
    tag = "" if (max_length == 4 and min_length == 1) else (f"_max{max_length}" if min_length == 1 else f"_min{min_length}-max{max_length}")

    for model in models:
        model_name = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        report_dir = os.path.join(repo_root, model_dir, "results", dataset, tokenizer)
        os.makedirs(report_dir, exist_ok=True)
        report_json = os.path.join(report_dir, f"strategy_comparison{tag}.json")

        existing_entries = {}
        existing_sid_entries = {}
        existing_pairwise_match = {}
        existing_pairwise_mae = {}
        if not args.no_merge and os.path.isfile(report_json):
            try:
                with open(report_json, encoding="utf-8") as f:
                    prev_raw = json.load(f)
                prev_list = (
                    prev_raw.get("strategy_comparison", prev_raw.get("recommendation_metrics", prev_raw))
                    if isinstance(prev_raw, dict)
                    else prev_raw
                )
                if isinstance(prev_list, list):
                    for item in prev_list:
                        if isinstance(item, dict) and "strategy" in item:
                            s_name = item["strategy"]
                            if s_name == "popularity":
                                s_name = "popularity:frequency"
                            existing_entries[s_name] = item

                if isinstance(prev_raw, dict):
                    sid_list = prev_raw.get("semantic_id_metrics", prev_raw.get("semantic_id_evaluation", []))
                    if isinstance(sid_list, list):
                        for item in sid_list:
                            if isinstance(item, dict) and "strategy" in item:
                                existing_sid_entries[item["strategy"]] = item
                    existing_pairwise_match = prev_raw.get("pairwise_exact_match_pct", {})
                    existing_pairwise_mae = prev_raw.get("pairwise_mae_tokens", {})
            except Exception:
                pass

        discovered_strategies = []
        if not args.no_merge and os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                s = file_to_strategy(fname, tag)
                if s and s not in discovered_strategies:
                    discovered_strategies.append(s)

        # Build comprehensive list of strategies
        has_collab_expanded = any(
            k.startswith("popularity:") and k != "popularity:frequency"
            for k in list(existing_entries.keys()) + discovered_strategies
        )
        combined_strategies = []
        for s in strategies:
            norm_s = "popularity:frequency" if (s == "popularity" and has_collab_expanded) else s
            if norm_s not in combined_strategies:
                combined_strategies.append(norm_s)

        for s in discovered_strategies:
            if s not in combined_strategies:
                combined_strategies.append(s)

        for s in existing_entries:
            if s not in combined_strategies:
                combined_strategies.append(s)

        # Sort canonically
        combined_strategies.sort(key=strategy_sort_key)

        # --- Evaluate Semantic IDs via compare_semantic_ids ---
        sid_eval = {}
        try:
            sid_eval = evaluate_semantic_ids(
                dataset=dataset,
                tokenizer=tokenizer,
                repo_root=repo_root,
                data_root=data_root,
                min_length=min_length,
                max_length=max_length,
                strategies=combined_strategies,
                pairwise=not args.no_pairwise_sid,
                existing_sid_entries=existing_sid_entries,
            )
        except Exception as e:
            print(f"[Warning] Semantic ID evaluation failed: {e}", file=sys.stderr)

        sid_results_map = {r["strategy"]: r for r in sid_eval.get("results", [])}
        items_count = sid_eval.get("items_count", None)
        total_traffic = sid_eval.get("total_traffic", None)

        # Recommendation and Semantic ID tables
        recom_table_data = []
        sid_table_data = []

        for strat in combined_strategies:
            res_file = None
            if strat == "fixed":
                res_fname = "fixed.json" if max_length == 4 else f"fixed_L{max_length}.json"
                res_file = os.path.join(report_dir, res_fname)
            elif strat == "shortest_unique":
                res_file = os.path.join(report_dir, f"varlen{tag}.json")
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
                    "co_occurrence": ("pop-cooccur", "-pop-cooccur"),
                    "cooccurrence": ("pop-cooccur", "-pop-cooccur"),
                    "cooccur": ("pop-cooccur", "-pop-cooccur"),
                    "cf_density": ("pop-cf", "-pop-cf"),
                }
                _, res_tag = sig_map.get(sig, (f"pop-{sig}", f"-pop-{sig}"))
                full_res_tag = res_tag if not tag else f"{res_tag}{tag}"
                res_file = os.path.join(report_dir, f"varlen{full_res_tag}.json")
            elif strat == "residual":
                res_tag = "-res" if not tag else f"-res{tag}"
                res_file = os.path.join(report_dir, f"varlen{res_tag}.json")

            # Recommendation metrics
            metrics = {}
            status = "pending"
            if res_file and os.path.isfile(res_file):
                try:
                    with open(res_file, encoding="utf-8") as f:
                        rdata = json.load(f)
                    metrics = rdata.get("mean_results", {})
                    status = "completed"
                except Exception:
                    status = "error"

            if strat in existing_entries:
                prev_row = existing_entries[strat]
                if status != "completed" and prev_row.get("status") == "completed":
                    metrics = prev_row.get("metrics", {})
                    status = "completed"

            # Semantic ID evaluation metrics
            sid_metric = sid_results_map.get(strat) or existing_sid_entries.get(strat, {})
            mean_l = sid_metric.get("mean_length")
            w_mean_l = sid_metric.get("traffic_weighted_length") or sid_metric.get("weighted_length")
            token_savings = sid_metric.get("token_savings_pct")
            collisions = sid_metric.get("collisions", 0)
            length_dist = sid_metric.get("length_distribution", {})
            rho = sid_metric.get("spearman_rho_vs_frequency")
            label = sid_metric.get("label", strat)
            basis = sid_metric.get("basis", "")

            # Fallbacks from existing entries if needed
            if mean_l is None and strat in existing_entries:
                mean_l = existing_entries[strat].get("mean_length")
            if w_mean_l is None and strat in existing_entries:
                w_mean_l = existing_entries[strat].get("weighted_length")
            if collisions == 0 and strat in existing_entries:
                collisions = existing_entries[strat].get("collisions", 0)
            if token_savings is None and mean_l is not None and max_length > 0:
                ref_len = w_mean_l if w_mean_l is not None else mean_l
                token_savings = (1.0 - ref_len / max_length) * 100.0

            recom_table_data.append({
                "strategy": strat,
                "mean_length": mean_l,
                "weighted_length": w_mean_l,
                "token_savings_pct": token_savings,
                "collisions": collisions,
                "metrics": metrics,
                "status": status,
            })

            sid_table_data.append({
                "strategy": strat,
                "label": label,
                "basis": basis,
                "mean_length": mean_l,
                "traffic_weighted_length": w_mean_l,
                "token_savings_pct": token_savings,
                "length_distribution": length_dist,
                "collisions": collisions,
                "spearman_rho_vs_frequency": rho,
            })

        # Pairwise matrices merging
        pairwise_match = sid_eval.get("pairwise_exact_match_pct", {})
        pairwise_mae = sid_eval.get("pairwise_mae_tokens", {})
        if existing_pairwise_match:
            for s1, m_dict in existing_pairwise_match.items():
                if s1 not in pairwise_match:
                    pairwise_match[s1] = dict(m_dict)
                else:
                    for s2, val in m_dict.items():
                        pairwise_match[s1].setdefault(s2, val)
        if existing_pairwise_mae:
            for s1, m_dict in existing_pairwise_mae.items():
                if s1 not in pairwise_mae:
                    pairwise_mae[s1] = dict(m_dict)
                else:
                    for s2, val in m_dict.items():
                        pairwise_mae[s1].setdefault(s2, val)

        # Locate fixed baseline metrics
        fixed_metrics = None
        for row in recom_table_data:
            if row["strategy"] == "fixed" and row["status"] == "completed":
                fixed_metrics = row["metrics"]
                break

        # -------------------------------------------------------------
        # 1. TEXT FORMATTING
        # -------------------------------------------------------------
        max_s_len = max([len(r["strategy"]) for r in recom_table_data] + [len("Strategy")])
        strat_col_w = max(24, max_s_len + 2)

        # Section 1: Recommendation Performance Table
        rec_fmt = f"| {{:<{strat_col_w}}} | {{:<8}} | {{:<10}} | {{:<6}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} |"
        rec_sep = f"+{'-' * (strat_col_w + 2)}+----------+------------+--------+----------+----------+----------+----------+----------+"
        rec_header = rec_fmt.format(
            "Strategy", "Mean L", "W-Mean L", "Coll.", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10"
        )
        total_rec_width = len(rec_sep)

        header_lines = [
            "\n" + "=" * total_rec_width,
            f" LETTER STRATEGY COMPARISON & SEMANTIC ID EVALUATION: {model_name} ({dataset})",
        ]
        if items_count and total_traffic:
            header_lines.append(
                f" Catalog Items: {items_count:,} | Total Interactions: {total_traffic:,} | Token Depth: [{min_length}, {max_length}]"
            )
        header_lines.extend([
            "=" * total_rec_width,
            "\n 1. Recommendation Performance",
            rec_sep,
            rec_header,
            rec_sep,
        ])

        rec_lines = list(header_lines)
        for row in recom_table_data:
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
                rec_lines.append(rec_fmt.format(strat, ml_str, wml_str, col_str, h1, h5, h10, n5, n10))

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
                    rec_lines.append(rec_fmt.format("  (delta vs fix)", d_ml, d_wml, "", dh1, dh5, dh10, dn5, dn10))
            else:
                status_tag = f"({row['status']})"
                rec_lines.append(rec_fmt.format(strat, ml_str, wml_str, col_str, status_tag, "-", "-", "-", "-"))
        rec_lines.append(rec_sep)

        # Section 2: Semantic ID Evaluation & Compression Table
        layer_keys = list(range(1, max_length + 1))
        layer_headers = [f"L={k}" for k in layer_keys]
        layer_widths = [max(6, len(h)) for h in layer_headers]
        layer_hdr_str = " | ".join(f"{h:>{w}}" for h, w in zip(layer_headers, layer_widths))
        layer_sep_str = "+".join("-" * (w + 2) for w in layer_widths)

        sid_fmt_header = f"| {{:<{strat_col_w}}} | {{:<12}} | {{:<13}} | {{:<13}} | {layer_hdr_str} | {{:<6}} | {{:<10}} |"
        sid_sep = f"+{'-' * (strat_col_w + 2)}+--------------+---------------+---------------+{layer_sep_str}+--------+------------+"
        sid_header = sid_fmt_header.format(
            "Strategy", "Catalog Mean", "Traffic W-Len", "Token Savings", "Coll.", "Spearman ρ"
        )
        sid_lines = [
            "\n 2. Semantic ID Evaluation & Compression",
            sid_sep,
            sid_header,
            sid_sep,
        ]
        for row in sid_table_data:
            strat = row["strategy"]
            ml_val = row["mean_length"]
            wml_val = row["traffic_weighted_length"]
            sav_val = row["token_savings_pct"]
            ml_str = f"{ml_val:12.3f}" if ml_val is not None else "           -"
            wml_str = f"{wml_val:13.3f}" if wml_val is not None else "            -"
            sav_str = f"{sav_val:12.1f}%" if sav_val is not None else "            -"
            d = row.get("length_distribution", {})
            tot = items_count or (sum(d.values()) if d else 0)
            if tot and tot > 0 and d and any(d.values()):
                layer_vals = [
                    f"{(d.get(k, d.get(str(k), 0)) / tot) * 100.0:.1f}%"
                    for k in layer_keys
                ]
            else:
                layer_vals = ["-" for _ in layer_keys]
            layer_val_str = " | ".join(f"{v:>{w}}" for v, w in zip(layer_vals, layer_widths))
            col_str = str(row["collisions"])
            rho_val = row["spearman_rho_vs_frequency"]
            rho_str = f"{rho_val:10.4f}" if rho_val is not None else "         -"
            sid_lines.append(
                f"| {strat:<{strat_col_w}} | {ml_str} | {wml_str} | {sav_str} | {layer_val_str} | {col_str:<6} | {rho_str} |"
            )
        sid_lines.append(sid_sep)

        # Sections 3 & 4: Pairwise Tables
        pairwise_text_sections = []
        valid_pairwise_strats = [s for s in combined_strategies if s in pairwise_match]
        if not args.no_pairwise_sid and len(valid_pairwise_strats) > 1:
            p_col_w = max(14, max(len(s) for s in valid_pairwise_strats + ["Strategy"]) + 2)
            p_row_fmt = f"| {{:<{p_col_w}}} | " + " | ".join([f"{{:<{p_col_w}}}" for _ in valid_pairwise_strats]) + " |"
            p_sep = f"+{'-' * (p_col_w + 2)}+" + "+".join([f"{'-' * (p_col_w + 2)}" for _ in valid_pairwise_strats]) + "+"
            p_width = len(p_sep)

            pairwise_text_sections.extend([
                "\n" + "=" * p_width,
                " 3. Pairwise Exact Semantic ID Agreement Rate (%)",
                " (Percentage of items in catalog assigned identical Semantic IDs between strategies)",
                "=" * p_width,
                p_row_fmt.format("Strategy", *valid_pairwise_strats),
                p_sep,
            ])
            for s1 in valid_pairwise_strats:
                row_vals = [
                    f"{pairwise_match.get(s1, {}).get(s2, 0.0):.2f}%" if s2 in pairwise_match.get(s1, {}) else "-"
                    for s2 in valid_pairwise_strats
                ]
                pairwise_text_sections.append(p_row_fmt.format(s1, *row_vals))
            pairwise_text_sections.append(p_sep)

            pairwise_text_sections.extend([
                "\n" + "=" * p_width,
                " 4. Pairwise Mean Absolute Length Difference (Tokens)",
                " (Average token length divergence per item: sum(|L_A - L_B|) / N)",
                "=" * p_width,
                p_row_fmt.format("Strategy", *valid_pairwise_strats),
                p_sep,
            ])
            for s1 in valid_pairwise_strats:
                row_vals = [
                    f"{pairwise_mae.get(s1, {}).get(s2, 0.0):.3f}" if s2 in pairwise_mae.get(s1, {}) else "-"
                    for s2 in valid_pairwise_strats
                ]
                pairwise_text_sections.append(p_row_fmt.format(s1, *row_vals))
            pairwise_text_sections.append(p_sep)

        full_text_lines = rec_lines + sid_lines + pairwise_text_sections
        summary_text = "\n".join(full_text_lines)
        print(summary_text)

        # -------------------------------------------------------------
        # 2. MARKDOWN FORMATTING
        # -------------------------------------------------------------
        tok_label = "Vanilla RQ-VAE" if tokenizer == "rqvae" else "LETTER"
        md_lines = [
            f"# {tok_label} Strategy Comparison Report: {model_name} ({dataset})\n",
            f"- **Dataset**: `{dataset}`",
            f"- **Model**: `{model_name}`",
            f"- **Tokenizer**: `{tokenizer}`",
        ]
        if items_count and total_traffic:
            md_lines.extend([
                f"- **Catalog Items**: {items_count:,}",
                f"- **Total Interactions**: {total_traffic:,}",
            ])
        md_lines.extend([
            f"- **Fixed ID Depth**: {max_length} tokens",
            f"- **Minimum Allowable Depth**: {min_length} token\n",
            "### 1. Recommendation Performance\n",
            "| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in recom_table_data:
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

        md_sid_header = (
            "| Strategy | Catalog Mean | Traffic W-Len | Token Savings | "
            + " | ".join(layer_headers)
            + " | Collisions | Spearman ρ |"
        )
        md_sid_sep = (
            "| :--- | :---: | :---: | :---: | "
            + " | ".join([":---:" for _ in layer_keys])
            + " | :---: | :---: |"
        )
        md_lines.extend([
            "",
            "### 2. Semantic ID Evaluation & Compression\n",
            md_sid_header,
            md_sid_sep,
        ])
        for row in sid_table_data:
            strat_name = row["strategy"]
            ml_val = row["mean_length"]
            wml_val = row["traffic_weighted_length"]
            sav_val = row["token_savings_pct"]
            ml = f"{ml_val:.3f}" if ml_val is not None else "-"
            wml = f"{wml_val:.3f}" if wml_val is not None else "-"
            sav = f"{sav_val:.1f}%" if sav_val is not None else "-"
            d = row.get("length_distribution", {})
            tot = items_count or (sum(d.values()) if d else 0)
            if tot and tot > 0 and d and any(d.values()):
                layer_vals = [
                    f"{(d.get(k, d.get(str(k), 0)) / tot) * 100.0:.1f}%"
                    for k in layer_keys
                ]
            else:
                layer_vals = ["-" for _ in layer_keys]
            col = str(row["collisions"])
            rho_val = row["spearman_rho_vs_frequency"]
            rho = f"{rho_val:.4f}" if rho_val is not None else "-"
            md_lines.append(
                f"| **{strat_name}** | {ml} | {wml} | {sav} | "
                + " | ".join(layer_vals)
                + f" | {col} | {rho} |"
            )

        if not args.no_pairwise_sid and len(valid_pairwise_strats) > 1:
            md_header = "| Strategy | " + " | ".join([f"{s}" for s in valid_pairwise_strats]) + " |"
            md_sep = "| :--- | " + " | ".join([":---:" for _ in valid_pairwise_strats]) + " |"

            md_lines.extend([
                "",
                "### 3. Pairwise Exact Semantic ID Agreement Rate (%)",
                "",
                "> Percentage of items in catalog that receive an identical Semantic ID across strategies.",
                "",
                md_header,
                md_sep,
            ])
            for s1 in valid_pairwise_strats:
                row_vals = [
                    f"{pairwise_match.get(s1, {}).get(s2, 0.0):.2f}%" if s2 in pairwise_match.get(s1, {}) else "-"
                    for s2 in valid_pairwise_strats
                ]
                md_lines.append(f"| **{s1}** | " + " | ".join(row_vals) + " |")

            md_lines.extend([
                "",
                "### 4. Pairwise Mean Absolute Length Difference (Tokens)",
                "",
                "> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.",
                "",
                md_header,
                md_sep,
            ])
            for s1 in valid_pairwise_strats:
                row_vals = [
                    f"{pairwise_mae.get(s1, {}).get(s2, 0.0):.3f}" if s2 in pairwise_mae.get(s1, {}) else "-"
                    for s2 in valid_pairwise_strats
                ]
                md_lines.append(f"| **{s1}** | " + " | ".join(row_vals) + " |")

        # Save Markdown report
        report_md = os.path.join(report_dir, f"strategy_comparison{tag}.md")
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved Markdown report to: {report_md}\n")


if __name__ == "__main__":
    main()

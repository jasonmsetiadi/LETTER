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
        description="Generate comparison report and tables across LETTER strategies and phases."
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
        "--phase",
        type=str,
        default="both",
        choices=["both", "all", "1", "1.5"],
        help="Training phase to report: 1 (Phase 1 post-hoc), 1.5 (Phase 1.5 length-aware), or both/all (default: both).",
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate rate-distortion Pareto frontier plot image.",
    )
    return parser.parse_args()


def file_to_strategy(filename, tag=""):
    """Infer (strategy, phase) tuple from result JSON filename."""
    if not filename.endswith(".json") or filename.startswith("strategy_comparison"):
        return None
    stem = filename[:-5]
    if tag:
        if not stem.endswith(tag):
            expected_fixed = f"fixed_L{tag.lstrip('_max')}" if tag.startswith("_max") else None
            if expected_fixed and stem == expected_fixed:
                return ("fixed", "fixed")
            if stem.startswith("fixed_L") and stem[7:].isdigit():
                return (stem, "fixed")
            return None
        stem = stem[:-len(tag)]
    else:
        if stem.startswith("fixed_L") and stem[7:].isdigit():
            return (stem, "fixed")
        if "_max" in stem or "_min" in stem:
            return None
    if stem == "fixed":
        return ("fixed", "fixed")
    if stem.startswith("fixed_L") and stem[7:].isdigit():
        return (stem, "fixed")

    phase = "1"
    if stem.endswith("_phase1.5") or stem.endswith("-phase1.5"):
        phase = "1.5"
        stem = stem[:-9]

    strat = None
    if stem == "varlen":
        strat = "shortest_unique"
    elif stem == "varlen-res":
        strat = "residual"
    elif stem == "varlen-pop":
        strat = "popularity:frequency"
    elif stem.startswith("varlen-pop-"):
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
            strat = f"popularity:{norm_sig}"

    if strat:
        return (strat, phase)
    return None


def strategy_to_filename(strat, phase, tag=""):
    """Construct expected result JSON filename for a (strategy, phase) pair."""
    if strat == "fixed":
        return "fixed.json" if not tag else (f"fixed_L{tag.lstrip('_max')}.json" if tag.startswith("_max") else f"fixed{tag}.json")
    if strat.startswith("fixed_L") and strat[7:].isdigit():
        return f"{strat}.json"
    phase_str = "_phase1.5" if phase == "1.5" else ""
    if strat == "shortest_unique":
        return f"varlen{phase_str}{tag}.json"
    if strat == "residual":
        return f"varlen-res{phase_str}{tag}.json"
    if strat == "popularity:frequency" or strat == "popularity":
        return f"varlen-pop{phase_str}{tag}.json"
    if strat.startswith("popularity:") or strat.startswith("collaborative:"):
        sig = strat.split(":", 1)[1]
        sig_map = {
            "frequency": "-pop",
            "user_entropy": "-pop-entropy",
            "pagerank": "-pop-pagerank",
            "co_occurrence": "-pop-cooccur",
            "cf_density": "-pop-cf",
        }
        prefix = sig_map.get(sig, f"-pop-{sig}")
        return f"varlen{prefix}{phase_str}{tag}.json"
    return None


def strategy_sort_key(strat_name, max_length=4):
    """Sort order: fixed (ordered by length) -> shortest_unique -> popularity signals -> residual -> others."""
    if strat_name == "fixed":
        return (0, max_length, 0, strat_name)
    if strat_name.startswith("fixed_L") and strat_name[7:].isdigit():
        return (0, int(strat_name[7:]), 0, strat_name)
    if strat_name == "shortest_unique":
        return (1, 0, 0, strat_name)
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
        return (2, collab_order.get(sig, 50), 0, strat_name)
    if strat_name == "residual":
        return (3, 0, 0, strat_name)
    return (4, 0, 0, strat_name)


def item_sort_key(item, max_length=4):
    """Sort order for (strategy, phase) items."""
    strat, phase = item if isinstance(item, tuple) else (item, "1")
    base = strategy_sort_key(strat, max_length=max_length)
    phase_order = 0 if phase in ("fixed", "-") else (1 if phase == "1" else 2)
    return (base[0], base[1], base[2], strat, phase_order)


def compute_rate_distortion_frontier(recom_table_data, max_length=4):
    """
    Computes Rate-Distortion Pareto Frontier from evaluated fixed baselines,
    and brackets all variable-length strategies against the frontier.
    """
    fixed_curve = []
    seen_lengths = set()

    for row in recom_table_data:
        strat = row["strategy"]
        phase = row["phase"]
        if phase in ("fixed", "-") or strat == "fixed" or strat.startswith("fixed_L"):
            if row.get("status") == "completed":
                if strat == "fixed":
                    d = float(row.get("mean_length") or max_length)
                elif strat.startswith("fixed_L") and strat[7:].isdigit():
                    d = float(strat[7:])
                else:
                    d = float(row.get("mean_length") or max_length)
                if d not in seen_lengths:
                    seen_lengths.add(d)
                    fixed_curve.append({
                        "length": d,
                        "strategy": strat,
                        "mean_length": d,
                        "metrics": row.get("metrics", {}),
                    })

    fixed_curve.sort(key=lambda x: x["length"])

    ref_fixed = None
    if fixed_curve:
        at_max = [f for f in fixed_curve if abs(f["length"] - max_length) < 1e-4]
        ref_fixed = at_max[0] if at_max else fixed_curve[-1]

    variable_bracketed = []
    pareto_dominant_strats = []

    for row in recom_table_data:
        strat = row["strategy"]
        phase = row["phase"]
        if phase in ("fixed", "-") or strat == "fixed" or strat.startswith("fixed_L"):
            continue

        mean_l = row.get("mean_length")
        w_mean_l = row.get("weighted_length")
        status = row.get("status")
        metrics = row.get("metrics", {})
        savings = row.get("token_savings_pct")

        entry = {
            "strategy": strat,
            "phase": phase,
            "mean_length": mean_l,
            "weighted_length": w_mean_l,
            "token_savings_pct": savings,
            "status": status,
            "metrics": metrics,
            "floor_length": None,
            "ceil_length": None,
            "floor_metrics": None,
            "ceil_metrics": None,
            "interp_metrics": {},
            "delta_interp": {},
            "delta_lmax": {},
            "delta_floor": {},
            "delta_ceil": {},
            "pareto_status": "pending",
        }

        if status != "completed" or mean_l is None or not fixed_curve:
            if status != "completed":
                entry["pareto_status"] = f"({status})"
            elif not fixed_curve:
                entry["pareto_status"] = "no fixed curve"
            variable_bracketed.append(entry)
            continue

        floors = [f for f in fixed_curve if f["length"] <= mean_l]
        ceilings = [f for f in fixed_curve if f["length"] >= mean_l]

        floor_pt = max(floors, key=lambda x: x["length"]) if floors else None
        ceil_pt = min(ceilings, key=lambda x: x["length"]) if ceilings else None

        if floor_pt:
            entry["floor_length"] = floor_pt["length"]
            entry["floor_metrics"] = floor_pt["metrics"]
        if ceil_pt:
            entry["ceil_length"] = ceil_pt["length"]
            entry["ceil_metrics"] = ceil_pt["metrics"]

        for m_key in ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]:
            val = metrics.get(m_key, 0.0)

            if ref_fixed:
                ref_val = ref_fixed["metrics"].get(m_key, 0.0)
                if ref_val > 0:
                    entry["delta_lmax"][m_key] = (val - ref_val) / ref_val * 100.0

            interp_val = None
            if floor_pt and ceil_pt:
                f_val = floor_pt["metrics"].get(m_key, 0.0)
                c_val = ceil_pt["metrics"].get(m_key, 0.0)
                if abs(ceil_pt["length"] - floor_pt["length"]) < 1e-6:
                    interp_val = c_val
                else:
                    alpha = (mean_l - floor_pt["length"]) / (ceil_pt["length"] - floor_pt["length"])
                    interp_val = (1.0 - alpha) * f_val + alpha * c_val
            elif floor_pt and not ceil_pt:
                interp_val = floor_pt["metrics"].get(m_key, 0.0)
            elif ceil_pt and not floor_pt:
                c_val = ceil_pt["metrics"].get(m_key, 0.0)
                interp_val = (mean_l / ceil_pt["length"]) * c_val if ceil_pt["length"] > 0 else c_val

            if interp_val is not None:
                entry["interp_metrics"][m_key] = interp_val
                if interp_val > 0:
                    entry["delta_interp"][m_key] = (val - interp_val) / interp_val * 100.0

            if floor_pt:
                f_val = floor_pt["metrics"].get(m_key, 0.0)
                if f_val > 0:
                    entry["delta_floor"][m_key] = (val - f_val) / f_val * 100.0

            if ceil_pt:
                c_val = ceil_pt["metrics"].get(m_key, 0.0)
                if c_val > 0:
                    entry["delta_ceil"][m_key] = (val - c_val) / c_val * 100.0

        v_ndcg = metrics.get("ndcg@10", 0.0)
        d_interp_ndcg = entry["delta_interp"].get("ndcg@10")
        c_ndcg = ceil_pt["metrics"].get("ndcg@10", 0.0) if ceil_pt else None

        if ceil_pt and c_ndcg and v_ndcg >= c_ndcg:
            entry["pareto_status"] = "★ Dominant"
            pareto_dominant_strats.append((strat, phase))
        elif d_interp_ndcg is not None and d_interp_ndcg > 0:
            entry["pareto_status"] = "▲ Efficient"
        elif d_interp_ndcg is not None and d_interp_ndcg >= -2.0:
            entry["pareto_status"] = "≈ Parity"
        elif d_interp_ndcg is not None:
            entry["pareto_status"] = "▼ Trade-off"
        elif ref_fixed:
            d_lmax = entry["delta_lmax"].get("ndcg@10", 0.0)
            if d_lmax >= 0:
                entry["pareto_status"] = "★ Dominant"
                pareto_dominant_strats.append((strat, phase))
            else:
                entry["pareto_status"] = "▼ Trade-off"

        variable_bracketed.append(entry)

    return {
        "fixed_curve": fixed_curve,
        "reference_fixed": ref_fixed,
        "variable_bracketed": variable_bracketed,
        "pareto_dominant_strats": pareto_dominant_strats,
    }


def generate_rate_distortion_plot(rd_data, dataset, model_name, out_png_path):
    """Generate a 2-panel publication-quality Rate-Distortion Pareto Frontier plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fixed_curve = rd_data.get("fixed_curve", [])
    var_list = rd_data.get("variable_bracketed", [])

    has_var_data = any(v.get("status") == "completed" for v in var_list)
    if not fixed_curve and not has_var_data:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    def plot_panel(ax, metric_key, metric_label):
        if fixed_curve:
            fx_lens = [f["length"] for f in fixed_curve]
            fx_vals = [f["metrics"].get(metric_key, 0.0) * 100.0 for f in fixed_curve]
            ax.plot(
                fx_lens, fx_vals,
                color="#2c3e50", marker="o", markersize=7, linewidth=2.2,
                label="Fixed Baseline Frontier", zorder=3
            )
            for x, y, f in zip(fx_lens, fx_vals, fixed_curve):
                lbl = f"L={int(x) if x.is_integer() else x}"
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color="#2c3e50", fontweight="bold")

        palette = {
            "shortest_unique": "#e67e22",
            "residual": "#27ae60",
            "popularity": "#2980b9",
            "collaborative": "#8e44ad",
        }

        for v in var_list:
            if v.get("status") != "completed" or v.get("mean_length") is None:
                continue
            x = v["mean_length"]
            y = v["metrics"].get(metric_key, 0.0) * 100.0
            strat = v["strategy"]
            phase = v["phase"]
            p_status = v.get("pareto_status", "")

            c = "#7f8c8d"
            for k, col in palette.items():
                if k in strat:
                    c = col
                    break

            marker = "*" if phase == "1.5" else "s"
            size = 140 if phase == "1.5" else 80
            alpha = 0.95 if phase == "1.5" else 0.8
            label_tag = f"{strat} (P{phase})" if phase not in ("fixed", "-") else strat

            ax.scatter(x, y, color=c, marker=marker, s=size, alpha=alpha,
                       edgecolors="black", linewidths=0.8, zorder=5, label=label_tag)

            interp_y = v.get("interp_metrics", {}).get(metric_key)
            if interp_y is not None:
                iy = interp_y * 100.0
                ax.plot([x, x], [iy, y], color=c, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)

            if "Dominant" in p_status:
                ax.scatter(x, y, s=size * 2.2, facecolors="none", edgecolors="#f39c12", linewidths=2.0, zorder=4)

        ax.set_title(f"Rate-Distortion Frontier: {metric_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Average Sequence Length (Tokens)", fontsize=10)
        ax.set_ylabel(f"{metric_label} (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plot_panel(ax1, "ndcg@10", "NDCG@10")
    plot_panel(ax2, "hit@10", "Hit@10")

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    try:
        fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[Warning] Failed to save plot: {e}", file=sys.stderr)
        plt.close(fig)
        return False


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

    target_phase = args.phase.lower()

    for model in models:
        model_name = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        report_dir = os.path.join(repo_root, model_dir, "results", dataset, tokenizer)
        if not os.path.isdir(report_dir) and tokenizer == "letter":
            legacy_dir = os.path.join(repo_root, model_dir, "results", dataset)
            if os.path.isdir(legacy_dir) and any(f.endswith(".json") for f in os.listdir(legacy_dir)):
                report_dir = legacy_dir
        os.makedirs(report_dir, exist_ok=True)

        if target_phase == "1.5":
            report_json = os.path.join(report_dir, f"strategy_comparison_phase1.5{tag}.json")
            report_md = os.path.join(report_dir, f"strategy_comparison_phase1.5{tag}.md")
        elif target_phase == "1":
            report_json = os.path.join(report_dir, f"strategy_comparison{tag}.json")
            report_md = os.path.join(report_dir, f"strategy_comparison{tag}.md")
        else:
            report_json = os.path.join(report_dir, f"strategy_comparison{tag}.json")
            report_md = os.path.join(report_dir, f"strategy_comparison{tag}.md")

        existing_entries = {}
        existing_sid_entries = {}
        existing_pairwise_match = {}
        existing_pairwise_mae = {}

        json_cands = [report_json]
        if target_phase in ("both", "all"):
            p15_json = os.path.join(report_dir, f"strategy_comparison_phase1.5{tag}.json")
            if p15_json != report_json and os.path.isfile(p15_json):
                json_cands.append(p15_json)

        if not args.no_merge:
            for jf in json_cands:
                if os.path.isfile(jf):
                    try:
                        with open(jf, encoding="utf-8") as f:
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
                                    s_ph = str(item.get("phase", "fixed" if s_name == "fixed" else "1"))
                                    existing_entries[(s_name, s_ph)] = item
                                    if s_name not in existing_entries:
                                        existing_entries[s_name] = item

                        if isinstance(prev_raw, dict):
                            sid_list = prev_raw.get("semantic_id_metrics", prev_raw.get("semantic_id_evaluation", []))
                            if isinstance(sid_list, list):
                                for item in sid_list:
                                    if isinstance(item, dict) and "strategy" in item:
                                        s_name = item["strategy"]
                                        if s_name == "popularity":
                                            s_name = "popularity:frequency"
                                        s_ph = str(item.get("phase", "fixed" if s_name == "fixed" else "1"))
                                        existing_sid_entries[(s_name, s_ph)] = item
                                        if s_name not in existing_sid_entries:
                                            existing_sid_entries[s_name] = item
                            p_match = prev_raw.get("pairwise_exact_match_pct", {})
                            for k, v in p_match.items():
                                existing_pairwise_match.setdefault(k, {}).update(v)
                            p_mae = prev_raw.get("pairwise_mae_tokens", {})
                            for k, v in p_mae.items():
                                existing_pairwise_mae.setdefault(k, {}).update(v)
                    except Exception:
                        pass

        discovered_items = []
        if not args.no_merge and os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                res = file_to_strategy(fname, tag)
                if res and res not in discovered_items:
                    discovered_items.append(res)

        def check_has_index(strat, ph):
            if ph != "1.5":
                return False
            tok_d = os.path.join(data_root, dataset, tokenizer)
            ds_d = os.path.join(data_root, dataset)
            var_t = "" if (max_length == 4 and min_length == 1) else (f".max{max_length}" if min_length == 1 else f".min{min_length}-max{max_length}")
            strat_sfx = ""
            if strat == "residual":
                strat_sfx = ".res"
            elif strat.startswith("popularity:") or strat.startswith("pop-") or strat.startswith("collaborative:"):
                sig = strat.split(":", 1)[1] if ":" in strat else strat[4:]
                sig_map = {
                    "frequency": ".pop", "raw": ".pop", "pop": ".pop",
                    "user_entropy": ".pop-entropy", "entropy": ".pop-entropy",
                    "pagerank": ".pop-pagerank", "pr": ".pop-pagerank",
                    "co_occurrence": ".pop-cooccur", "cooccur": ".pop-cooccur",
                    "cf_density": ".pop-cf",
                }
                strat_sfx = sig_map.get(sig, f".pop-{sig}")
            cands = [
                os.path.join(tok_d, f"{dataset}.index.varlen{strat_sfx}-phase1.5{var_t}.json"),
                os.path.join(tok_d, f"{dataset}.index.varlen{strat_sfx}-phase1.5.json"),
                os.path.join(ds_d, f"{dataset}.index.varlen{strat_sfx}-phase1.5{var_t}.json"),
                os.path.join(ds_d, f"{dataset}.index.varlen{strat_sfx}-phase1.5.json"),
            ]
            return any(os.path.isfile(c) for c in cands)

        has_collab_expanded = any(
            isinstance(k, tuple) and k[0].startswith("popularity:") and k[0] != "popularity:frequency"
            for k in list(existing_entries.keys()) + discovered_items
        )

        combined_items = []

        def add_entry(s_name, s_phase):
            norm_s = "popularity:frequency" if (s_name == "popularity" and has_collab_expanded) else s_name
            if target_phase == "1" and s_phase == "1.5":
                return
            if target_phase == "1.5" and s_phase == "1":
                return
            pair = (norm_s, s_phase)
            if pair not in combined_items:
                combined_items.append(pair)

        add_entry("fixed", "fixed")

        for s in strategies:
            if s == "fixed":
                continue
            if s.startswith("fixed_L"):
                add_entry(s, "fixed")
                continue
            if target_phase in ("both", "all", "1"):
                add_entry(s, "1")
            if target_phase == "1.5":
                add_entry(s, "1.5")
            elif target_phase in ("both", "all"):
                if (s, "1.5") in discovered_items or (s, "1.5") in existing_entries or check_has_index(s, "1.5"):
                    add_entry(s, "1.5")

        for itm in discovered_items:
            add_entry(itm[0], itm[1])

        for k in existing_entries:
            if isinstance(k, tuple):
                add_entry(k[0], k[1])

        if ("fixed", "fixed") in combined_items and (f"fixed_L{max_length}", "fixed") in combined_items:
            combined_items.remove((f"fixed_L{max_length}", "fixed"))

        combined_items.sort(key=lambda it: item_sort_key(it, max_length=max_length))

        # Evaluate Semantic IDs
        sid_eval = {}
        try:
            sid_eval = evaluate_semantic_ids(
                dataset=dataset,
                tokenizer=tokenizer,
                repo_root=repo_root,
                data_root=data_root,
                min_length=min_length,
                max_length=max_length,
                strategies=combined_items,
                pairwise=not args.no_pairwise_sid,
                existing_sid_entries=existing_sid_entries,
            )
        except Exception as e:
            print(f"[Warning] Semantic ID evaluation failed: {e}", file=sys.stderr)

        sid_results_map = {}
        for r in sid_eval.get("results", []):
            st = r["strategy"]
            ph = str(r.get("phase", "1"))
            sid_results_map[(st, ph)] = r
            if st not in sid_results_map:
                sid_results_map[st] = r

        items_count = sid_eval.get("items_count", None)
        total_traffic = sid_eval.get("total_traffic", None)

        recom_table_data = []
        sid_table_data = []

        for strat, phase in combined_items:
            res_fname = strategy_to_filename(strat, phase, tag)
            res_file = os.path.join(report_dir, res_fname) if res_fname else None
            if strat == "fixed" and (not res_file or not os.path.isfile(res_file)):
                alt = os.path.join(report_dir, f"fixed_L{max_length}.json")
                if os.path.isfile(alt):
                    res_file = alt
            elif strat.startswith("fixed_L") and strat[7:].isdigit() and int(strat[7:]) == max_length and (not res_file or not os.path.isfile(res_file)):
                alt = os.path.join(report_dir, "fixed.json")
                if os.path.isfile(alt):
                    res_file = alt

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

            prev_row = existing_entries.get((strat, phase)) or existing_entries.get(strat)
            if prev_row:
                if status != "completed" and prev_row.get("status") == "completed":
                    metrics = prev_row.get("metrics", {})
                    status = "completed"

            sid_metric = (
                sid_results_map.get((strat, phase))
                or sid_results_map.get(strat)
                or existing_sid_entries.get((strat, phase))
                or existing_sid_entries.get(strat, {})
            )
            mean_l = sid_metric.get("mean_length")
            w_mean_l = sid_metric.get("traffic_weighted_length") or sid_metric.get("weighted_length")
            token_savings = sid_metric.get("token_savings_pct")
            collisions = sid_metric.get("collisions", 0)
            length_dist = sid_metric.get("length_distribution", {})
            rho = sid_metric.get("spearman_rho_vs_frequency")
            label = sid_metric.get("label", strat)
            basis = sid_metric.get("basis", "")

            if mean_l is None and prev_row:
                mean_l = prev_row.get("mean_length")
            if w_mean_l is None and prev_row:
                w_mean_l = prev_row.get("weighted_length")
            if collisions == 0 and prev_row:
                collisions = prev_row.get("collisions", 0)

            if strat == "fixed" and mean_l is None:
                mean_l = float(max_length)
                w_mean_l = float(max_length)
                token_savings = 0.0
            elif strat.startswith("fixed_L") and strat[7:].isdigit() and mean_l is None:
                d_val = float(strat[7:])
                mean_l = d_val
                w_mean_l = d_val
                token_savings = ((max_length - d_val) / max_length * 100.0) if max_length > 0 else 0.0
            elif token_savings is None and mean_l is not None and max_length > 0:
                ref_len = w_mean_l if w_mean_l is not None else mean_l
                token_savings = (1.0 - ref_len / max_length) * 100.0

            recom_table_data.append({
                "strategy": strat,
                "phase": phase,
                "mean_length": mean_l,
                "weighted_length": w_mean_l,
                "token_savings_pct": token_savings,
                "collisions": collisions,
                "metrics": metrics,
                "status": status,
            })

            sid_table_data.append({
                "strategy": strat,
                "phase": phase,
                "label": label,
                "basis": basis,
                "mean_length": mean_l,
                "traffic_weighted_length": w_mean_l,
                "token_savings_pct": token_savings,
                "length_distribution": length_dist,
                "collisions": collisions,
                "spearman_rho_vs_frequency": rho,
            })

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

        # Compute Rate-Distortion Pareto Frontier
        rd_data = compute_rate_distortion_frontier(recom_table_data, max_length=max_length)
        fixed_curve = rd_data["fixed_curve"]
        ref_fixed = rd_data["reference_fixed"]
        fixed_metrics = ref_fixed["metrics"] if ref_fixed else None
        variable_bracketed = rd_data["variable_bracketed"]
        var_by_strat_ph = {(v["strategy"], v["phase"]): v for v in variable_bracketed}

        # Identify head-to-head strategies
        p1_by_strat = {r["strategy"]: r for r in recom_table_data if r["phase"] == "1"}
        p15_by_strat = {r["strategy"]: r for r in recom_table_data if r["phase"] == "1.5"}
        common_h2h = [s for s in p1_by_strat if s in p15_by_strat]

        # -------------------------------------------------------------
        # 1. TEXT FORMATTING
        # -------------------------------------------------------------
        max_s_len = max([len(r["strategy"]) for r in recom_table_data] + [len("Strategy")])
        strat_col_w = max(24, max_s_len + 2)

        # Section 1: Recommendation Performance Table
        rec_fmt = f"| {{:<{strat_col_w}}} | {{:^7}} | {{:<8}} | {{:<10}} | {{:<6}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} |"
        rec_sep = f"+{'-' * (strat_col_w + 2)}+---------+----------+------------+--------+----------+----------+----------+----------+----------+"
        rec_header = rec_fmt.format(
            "Strategy", "Phase", "Mean L", "W-Mean L", "Coll.", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10"
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
            phase_str = "-" if row["phase"] in ("fixed", "-") else row["phase"]
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
                rec_lines.append(rec_fmt.format(strat, phase_str, ml_str, wml_str, col_str, h1, h5, h10, n5, n10))

                if strat != "fixed" and not strat.startswith("fixed_L") and fixed_metrics:
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
                    rec_lines.append(rec_fmt.format("  (delta vs fix)", "", d_ml, d_wml, "", dh1, dh5, dh10, dn5, dn10))

                    v_brk = var_by_strat_ph.get((strat, row["phase"]))
                    if v_brk and v_brk.get("delta_interp"):
                        di = v_brk["delta_interp"]
                        d_ih1 = f"{di['hit@1']:+.1f}%" if "hit@1" in di else "-"
                        d_ih5 = f"{di['hit@5']:+.1f}%" if "hit@5" in di else "-"
                        d_ih10 = f"{di['hit@10']:+.1f}%" if "hit@10" in di else "-"
                        d_in5 = f"{di['ndcg@5']:+.1f}%" if "ndcg@5" in di else "-"
                        d_in10 = f"{di['ndcg@10']:+.1f}%" if "ndcg@10" in di else "-"
                        rec_lines.append(rec_fmt.format("  (vs interp-rate)", "", f"L={ml_val:.2f}" if ml_val is not None else "", "", "", d_ih1, d_ih5, d_ih10, d_in5, d_in10))
            else:
                status_tag = f"({row['status']})"
                rec_lines.append(rec_fmt.format(strat, phase_str, ml_str, wml_str, col_str, status_tag, "-", "-", "-", "-"))
        rec_lines.append(rec_sep)

        # Section 2: Semantic ID Evaluation & Compression Table
        layer_keys = list(range(1, max_length + 1))
        layer_headers = [f"L={k}" for k in layer_keys]
        layer_widths = [max(6, len(h)) for h in layer_headers]
        layer_hdr_str = " | ".join(f"{h:>{w}}" for h, w in zip(layer_headers, layer_widths))
        layer_sep_str = "+".join("-" * (w + 2) for w in layer_widths)

        sid_fmt_header = f"| {{:<{strat_col_w}}} | {{:^7}} | {{:<12}} | {{:<13}} | {{:<13}} | {layer_hdr_str} | {{:<6}} | {{:<10}} |"
        sid_sep = f"+{'-' * (strat_col_w + 2)}+---------+--------------+---------------+---------------+{layer_sep_str}+--------+------------+"
        sid_header = sid_fmt_header.format(
            "Strategy", "Phase", "Catalog Mean", "Traffic W-Len", "Token Savings", "Coll.", "Spearman ρ"
        )
        sid_lines = [
            "\n 2. Semantic ID Evaluation & Compression",
            sid_sep,
            sid_header,
            sid_sep,
        ]
        for row in sid_table_data:
            strat = row["strategy"]
            phase_str = "-" if row["phase"] in ("fixed", "-") else row["phase"]
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
                f"| {strat:<{strat_col_w}} | {phase_str:^7} | {ml_str} | {wml_str} | {sav_str} | {layer_val_str} | {col_str:<6} | {rho_str} |"
            )
        sid_lines.append(sid_sep)

        # Section 3: Rate-Distortion & Pareto Frontier Analysis Table
        rd_text_sections = []
        if variable_bracketed:
            rd_strat_w = max(24, max([len(v["strategy"]) for v in variable_bracketed] + [len("Strategy")]) + 2)
            rd_fmt = f"| {{:<{rd_strat_w}}} | {{:^7}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<8}} | {{:<11}} | {{:<11}} | {{:<13}} |"
            rd_sep = f"+{'-' * (rd_strat_w + 2)}+---------+----------+----------+----------+----------+----------+-------------+-------------+---------------+"

            fixed_summary_str = ""
            if fixed_curve:
                pts = [f"L={int(f['length']) if f['length'].is_integer() else f['length']} (H@10={f['metrics'].get('hit@10', 0)*100:.1f}%, N@10={f['metrics'].get('ndcg@10', 0)*100:.1f}%)" for f in fixed_curve]
                fixed_summary_str = " Frontier points: " + ", ".join(pts)

            rd_text_sections.extend([
                "\n" + "=" * len(rd_sep),
                " 3. Rate-Distortion & Pareto Frontier Analysis",
                " (Benchmarking variable-length token efficiency against matched-budget fixed baselines)",
            ])
            if fixed_summary_str:
                rd_text_sections.append(fixed_summary_str)
            rd_text_sections.extend([
                "=" * len(rd_sep),
                rd_sep,
                rd_fmt.format("Strategy", "Phase", "Mean L", "Hit@10", "NDCG@10", "Floor L", "Ceil L", "Interp N@10", "Δ vs Interp", "Pareto Status"),
                rd_sep,
            ])
            for v in variable_bracketed:
                s = v["strategy"]
                ph = "-" if v["phase"] in ("fixed", "-") else v["phase"]
                ml_s = f"{v['mean_length']:.2f}" if v['mean_length'] is not None else "-"
                m = v.get("metrics", {})
                if v.get("status") == "completed":
                    h10_s = f"{m.get('hit@10', 0)*100:.2f}%"
                    n10_s = f"{m.get('ndcg@10', 0)*100:.2f}%"
                    fl_s = f"L={int(v['floor_length']) if v['floor_length'].is_integer() else v['floor_length']}" if v.get("floor_length") is not None else "-"
                    cl_s = f"L={int(v['ceil_length']) if v['ceil_length'].is_integer() else v['ceil_length']}" if v.get("ceil_length") is not None else "-"
                    int_n10 = v.get("interp_metrics", {}).get("ndcg@10")
                    int_s = f"{int_n10*100:.2f}%" if int_n10 is not None else "-"
                    d_int = v.get("delta_interp", {}).get("ndcg@10")
                    d_int_s = f"{d_int:+.1f}%" if d_int is not None else "-"
                    p_stat = v.get("pareto_status", "-")
                    rd_text_sections.append(rd_fmt.format(s, ph, ml_s, h10_s, n10_s, fl_s, cl_s, int_s, d_int_s, p_stat))
                else:
                    rd_text_sections.append(rd_fmt.format(s, ph, ml_s, "-", "-", "-", "-", "-", "-", f"({v['status']})"))
            rd_text_sections.append(rd_sep)

        # Section 4: Head-to-Head Comparison (if applicable)
        head_to_head_text_sections = []
        if common_h2h:
            h2h_strat_w = max(24, max(len(s) for s in common_h2h + ["Strategy"]) + 2)
            h2h_fmt = f"| {{:<{h2h_strat_w}}} | {{:<9}} | {{:<10}} | {{:<9}} | {{:<10}} | {{:<9}} | {{:<9}} | {{:<10}} | {{:<9}} |"
            h2h_sep = f"+{'-' * (h2h_strat_w + 2)}+-----------+------------+-----------+------------+-----------+-----------+------------+-----------+"
            head_to_head_text_sections.extend([
                "\n" + "=" * len(h2h_sep),
                " 4. Phase 1 vs. Phase 1.5 Head-to-Head Comparison (Length-Aware Training Impact)",
                " (Direct delta: Phase 1.5 - Phase 1; positive values indicate Phase 1.5 gain)",
                "=" * len(h2h_sep),
                h2h_sep,
                h2h_fmt.format("Strategy", "P1 MeanL", "P1.5 MeanL", "P1 Hit@10", "P1.5 H@10", "Δ Hit@10", "P1 NDCG@10", "P1.5 N@10", "Δ NDCG@10"),
                h2h_sep,
            ])
            for s in common_h2h:
                r1 = p1_by_strat[s]
                r15 = p15_by_strat[s]
                m1 = r1.get("metrics", {})
                m15 = r15.get("metrics", {})
                l1_s = f"{r1['mean_length']:.2f}" if r1['mean_length'] is not None else "-"
                l15_s = f"{r15['mean_length']:.2f}" if r15['mean_length'] is not None else "-"
                if r1.get("status") == "completed" and r15.get("status") == "completed":
                    h10_1 = m1.get("hit@10", 0) * 100
                    h10_15 = m15.get("hit@10", 0) * 100
                    dh10 = h10_15 - h10_1
                    n10_1 = m1.get("ndcg@10", 0) * 100
                    n10_15 = m15.get("ndcg@10", 0) * 100
                    dn10 = n10_15 - n10_1
                    head_to_head_text_sections.append(
                        h2h_fmt.format(
                            s,
                            l1_s,
                            l15_s,
                            f"{h10_1:.2f}%",
                            f"{h10_15:.2f}%",
                            f"{dh10:+.2f}%",
                            f"{n10_1:.2f}%",
                            f"{n10_15:.2f}%",
                            f"{dn10:+.2f}%",
                        )
                    )
                else:
                    head_to_head_text_sections.append(
                        h2h_fmt.format(s, l1_s, l15_s, "-", "-", "-", "-", "-", "-")
                    )
            head_to_head_text_sections.append(h2h_sep)

        # Sections 5 & 6: Pairwise Tables
        pairwise_text_sections = []
        valid_pairwise_strats = list(pairwise_match.keys())
        p_sec_num1 = 5 if common_h2h else 4
        p_sec_num2 = 6 if common_h2h else 5

        if not args.no_pairwise_sid and len(valid_pairwise_strats) > 1:
            p_col_w = max(14, max(len(s) for s in valid_pairwise_strats + ["Strategy"]) + 2)
            p_row_fmt = f"| {{:<{p_col_w}}} | " + " | ".join([f"{{:<{p_col_w}}}" for _ in valid_pairwise_strats]) + " |"
            p_sep = f"+{'-' * (p_col_w + 2)}+" + "+".join([f"{'-' * (p_col_w + 2)}" for _ in valid_pairwise_strats]) + "+"
            p_width = len(p_sep)

            pairwise_text_sections.extend([
                "\n" + "=" * p_width,
                f" {p_sec_num1}. Pairwise Exact Semantic ID Agreement Rate (%)",
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
                f" {p_sec_num2}. Pairwise Mean Absolute Length Difference (Tokens)",
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

        full_text_lines = rec_lines + sid_lines + rd_text_sections + head_to_head_text_sections + pairwise_text_sections
        summary_text = "\n".join(full_text_lines)
        print(summary_text)

        # -------------------------------------------------------------
        # 2. MARKDOWN FORMATTING
        # -------------------------------------------------------------
        tok_label = "Vanilla RQ-VAE" if tokenizer == "rqvae" else "LETTER"
        phase_label = "Phase 1 & Phase 1.5 (All)" if target_phase in ("both", "all") else f"Phase {target_phase}"
        md_lines = [
            f"# {tok_label} Strategy Comparison Report: {model_name} ({dataset})\n",
            f"- **Dataset**: `{dataset}`",
            f"- **Model**: `{model_name}`",
            f"- **Tokenizer**: `{tokenizer}`",
            f"- **Training Phase**: `{phase_label}`",
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
            "| Strategy | Phase | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in recom_table_data:
            m = row["metrics"]
            phase_str = "-" if row["phase"] in ("fixed", "-") else row["phase"]
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
                md_lines.append(f"| **{strat_name}** | {phase_str} | {ml} | {wml} | {col} | {h1} | {h5} | {h10} | {n5} | {n10} | {st} |")
            else:
                md_lines.append(f"| **{strat_name}** | {phase_str} | {ml} | {wml} | {col} | - | - | - | - | - | {st} |")

        md_sid_header = (
            "| Strategy | Phase | Catalog Mean | Traffic W-Len | Token Savings | "
            + " | ".join(layer_headers)
            + " | Collisions | Spearman ρ |"
        )
        md_sid_sep = (
            "| :--- | :---: | :---: | :---: | :---: | "
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
            phase_str = "-" if row["phase"] in ("fixed", "-") else row["phase"]
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
                f"| **{strat_name}** | {phase_str} | {ml} | {wml} | {sav} | "
                + " | ".join(layer_vals)
                + f" | {col} | {rho} |"
            )

        # Plot generation
        plot_saved = False
        plot_fname = f"rate_distortion_frontier_{dataset}{tag}.png"
        plot_png_path = os.path.join(report_dir, plot_fname)
        if not args.no_plot:
            plot_saved = generate_rate_distortion_plot(
                rd_data=rd_data,
                dataset=dataset,
                model_name=model_name,
                out_png_path=plot_png_path,
            )

        md_sec_idx = 3
        if variable_bracketed:
            md_lines.extend([
                "",
                f"### {md_sec_idx}. Rate-Distortion & Pareto Frontier Analysis",
                "",
                "> Benchmarking variable-length token efficiency against fixed-length baselines of equivalent token budgets.",
                "> - **Floor Baseline ($L_{\\text{floor}}$)**: Highest evaluated fixed depth $\\le$ Mean Length.",
                "> - **Ceiling Baseline ($L_{\\text{ceil}}$)**: Lowest evaluated fixed depth $\\ge$ Mean Length.",
                "> - **Interpolated ($L_{\\text{interp}}$)**: Expected fixed metric at matched average token length: $M_{\\text{interp}} = (1-\\alpha) M_{\\text{floor}} + \\alpha M_{\\text{ceil}}$.",
                "> - **Pareto Classification**:",
                ">   - **★ Dominant**: Variable model achieves equal or higher recommendation quality using strictly fewer tokens than $L_{\\text{ceil}}$ ($M \\ge M_{\\text{ceil}}$ with $\\bar{L} \\le L_{\\text{ceil}}$).",
                ">   - **▲ Efficient**: Variable model outperforms the interpolated fixed baseline at matched token budget ($\\Delta_{\\text{interp}} > 0$).",
                ">   - **≈ Parity**: Within 2% of matched-budget baseline.",
                ">   - **▼ Trade-off**: Standard compression trade-off.",
                "",
            ])
            md_sec_idx += 1

            if plot_saved:
                md_lines.extend([
                    f"![Rate-Distortion Pareto Frontier]({plot_fname})\n",
                ])

            if len(fixed_curve) > 1:
                md_lines.extend([
                    "#### Reference Fixed-Length Rate-Distortion Curve\n",
                    "| Depth (L) | Mean Length | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Token Savings |",
                    "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ])
                for fc in fixed_curve:
                    f_d = fc["length"]
                    f_m = fc.get("metrics", {})
                    h1_str = f"{f_m.get('hit@1', 0)*100:.2f}%" if f_m.get('hit@1') is not None else "-"
                    h5_str = f"{f_m.get('hit@5', 0)*100:.2f}%" if f_m.get('hit@5') is not None else "-"
                    h10_str = f"{f_m.get('hit@10', 0)*100:.2f}%" if f_m.get('hit@10') is not None else "-"
                    n5_str = f"{f_m.get('ndcg@5', 0)*100:.2f}%" if f_m.get('ndcg@5') is not None else "-"
                    n10_str = f"{f_m.get('ndcg@10', 0)*100:.2f}%" if f_m.get('ndcg@10') is not None else "-"
                    sav_pct = ((max_length - f_d) / max_length * 100.0) if max_length > 0 else 0.0
                    md_lines.append(f"| **L={int(f_d) if f_d.is_integer() else f_d}** | {f_d:.2f} | {h1_str} | {h5_str} | {h10_str} | {n5_str} | {n10_str} | {sav_pct:.1f}% |")
                md_lines.append("")

            md_lines.extend([
                "#### Variable-Length Rate-Distortion Frontier & Pareto Evaluation\n",
                "| Strategy | Phase | Mean Length | Token Savings | Hit@10 | NDCG@10 | Floor L | Ceil L | Interp NDCG@10 | Δ vs Interp | Δ vs L_max | Pareto Status |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for v in variable_bracketed:
                s_name = v["strategy"]
                phase_str = "-" if v["phase"] in ("fixed", "-") else v["phase"]
                ml_s = f"{v['mean_length']:.2f}" if v['mean_length'] is not None else "-"
                sav_s = f"{v['token_savings_pct']:.1f}%" if v.get('token_savings_pct') is not None else "-"
                m = v.get("metrics", {})
                if v.get("status") == "completed":
                    h10_s = f"{m.get('hit@10', 0)*100:.2f}%"
                    n10_s = f"{m.get('ndcg@10', 0)*100:.2f}%"
                    fl_s = f"L={int(v['floor_length']) if v['floor_length'].is_integer() else v['floor_length']}" if v.get("floor_length") is not None else "-"
                    cl_s = f"L={int(v['ceil_length']) if v['ceil_length'].is_integer() else v['ceil_length']}" if v.get("ceil_length") is not None else "-"
                    int_n10 = v.get("interp_metrics", {}).get("ndcg@10")
                    int_s = f"{int_n10*100:.2f}%" if int_n10 is not None else "-"
                    d_int = v.get("delta_interp", {}).get("ndcg@10")
                    d_int_s = f"**{d_int:+.1f}%**" if d_int is not None else "-"
                    d_lm = v.get("delta_lmax", {}).get("ndcg@10")
                    d_lm_s = f"{d_lm:+.1f}%" if d_lm is not None else "-"
                    p_stat = v.get("pareto_status", "-")
                    md_lines.append(f"| **{s_name}** | {phase_str} | {ml_s} | {sav_s} | {h10_s} | {n10_s} | {fl_s} | {cl_s} | {int_s} | {d_int_s} | {d_lm_s} | {p_stat} |")
                else:
                    md_lines.append(f"| **{s_name}** | {phase_str} | {ml_s} | {sav_s} | - | - | - | - | - | - | - | ({v['status']}) |")

        if common_h2h:
            md_lines.extend([
                "",
                f"### {md_sec_idx}. Phase 1 vs. Phase 1.5 Head-to-Head Comparison",
                "",
                "> Direct comparison of Post-Hoc Truncation (Phase 1) vs. Length-Aware Training (Phase 1.5). Positive Δ indicates improvement from length-aware training.",
                "",
                "| Strategy | P1 Mean Length | P1.5 Mean Length | P1 Hit@10 | P1.5 Hit@10 | Δ Hit@10 | P1 NDCG@10 | P1.5 NDCG@10 | Δ NDCG@10 |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for s in common_h2h:
                r1 = p1_by_strat[s]
                r15 = p15_by_strat[s]
                m1 = r1.get("metrics", {})
                m15 = r15.get("metrics", {})
                l1_s = f"{r1['mean_length']:.2f}" if r1['mean_length'] is not None else "-"
                l15_s = f"{r15['mean_length']:.2f}" if r15['mean_length'] is not None else "-"
                if r1.get("status") == "completed" and r15.get("status") == "completed":
                    h10_1 = m1.get("hit@10", 0) * 100
                    h10_15 = m15.get("hit@10", 0) * 100
                    dh10 = h10_15 - h10_1
                    n10_1 = m1.get("ndcg@10", 0) * 100
                    n10_15 = m15.get("ndcg@10", 0) * 100
                    dn10 = n10_15 - n10_1
                    md_lines.append(
                        f"| **{s}** | {l1_s} | {l15_s} | {h10_1:.2f}% | {h10_15:.2f}% | **{dh10:+.2f}%** | {n10_1:.2f}% | {n10_15:.2f}% | **{dn10:+.2f}%** |"
                    )
                else:
                    md_lines.append(f"| **{s}** | {l1_s} | {l15_s} | - | - | - | - | - | - |")
            md_sec_idx += 1

        if not args.no_pairwise_sid and len(valid_pairwise_strats) > 1:
            md_header = "| Strategy | " + " | ".join([f"{s}" for s in valid_pairwise_strats]) + " |"
            md_sep = "| :--- | " + " | ".join([":---:" for _ in valid_pairwise_strats]) + " |"

            md_lines.extend([
                "",
                f"### {md_sec_idx}. Pairwise Exact Semantic ID Agreement Rate (%)",
                "",
                "> Percentage of items in catalog that receive an identical Semantic ID across strategies.",
                "",
                md_header,
                md_sep,
            ])
            md_sec_idx += 1
            for s1 in valid_pairwise_strats:
                row_vals = [
                    f"{pairwise_match.get(s1, {}).get(s2, 0.0):.2f}%" if s2 in pairwise_match.get(s1, {}) else "-"
                    for s2 in valid_pairwise_strats
                ]
                md_lines.append(f"| **{s1}** | " + " | ".join(row_vals) + " |")

            md_lines.extend([
                "",
                f"### {md_sec_idx}. Pairwise Mean Absolute Length Difference (Tokens)",
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

        # Save Markdown and JSON report
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"\nSaved Markdown report to: {report_md}\n")

        json_output = {
            "dataset": dataset,
            "model": model_name,
            "phase": target_phase,
            "catalog_items": items_count,
            "total_traffic": total_traffic,
            "min_length": min_length,
            "max_length": max_length,
            "fixed_curve": rd_data.get("fixed_curve", []),
            "rate_distortion_frontier": rd_data.get("variable_bracketed", []),
            "pareto_dominant_strategies": rd_data.get("pareto_dominant_strats", []),
            "strategy_comparison": recom_table_data,
            "semantic_id_metrics": sid_table_data,
            "pairwise_exact_match_pct": pairwise_match,
            "pairwise_mae_tokens": pairwise_mae,
        }
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved JSON report to: {report_json}\n")


if __name__ == "__main__":
    main()

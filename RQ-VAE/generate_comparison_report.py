import argparse
import json
import os
from pathlib import Path


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
    return parser.parse_args()


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

        table_data = []
        fixed_metrics = None

        for strat in strategies:
            idx_summary_file = None
            res_file = None

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
                # Fallback: inspect index file directly if summary file is missing
                actual_idx_name = (
                    f"{dataset}.index.varlen{tag}.json"
                    if strat == "shortest_unique"
                    else (
                        f"{dataset}.index.varlen.pop{tag}.json"
                        if strat == "popularity"
                        else f"{dataset}.index.varlen.res{tag}.json"
                    )
                )
                actual_idx_file = os.path.join(data_root, dataset, actual_idx_name)
                if os.path.isfile(actual_idx_file):
                    try:
                        with open(actual_idx_file) as f:
                            idata = json.load(f)
                        lens = [len(v) for v in idata.values()]
                        mean_l = sum(lens) / max(1, len(lens))
                        uniq = len({tuple(v) for v in idata.values()})
                        collisions = len(idata) - uniq
                    except Exception:
                        pass

            if w_mean_l is None and strat != "fixed":
                actual_idx_name = (
                    f"{dataset}.index.varlen{tag}.json"
                    if strat == "shortest_unique"
                    else (
                        f"{dataset}.index.varlen.pop{tag}.json"
                        if strat == "popularity"
                        else f"{dataset}.index.varlen.res{tag}.json"
                    )
                )
                idx_path = os.path.join(data_root, dataset, actual_idx_name)
                inter_path = os.path.join(data_root, dataset, f"{dataset}.inter.json")
                if os.path.isfile(idx_path) and os.path.isfile(inter_path):
                    try:
                        from collections import Counter
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
                    if strat == "fixed":
                        fixed_metrics = metrics
                except Exception:
                    status = "error"

            table_data.append({
                "strategy": strat,
                "mean_length": mean_l,
                "weighted_length": w_mean_l,
                "collisions": collisions,
                "metrics": metrics,
                "status": status,
            })

        row_fmt = "| {:<16} | {:<8} | {:<10} | {:<6} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
        sep = "+------------------+----------+------------+--------+----------+----------+----------+----------+----------+"
        header = row_fmt.format(
            "Strategy", "Mean L", "W-Mean L", "Coll.", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10"
        )

        lines = [
            "\n" + "=" * 99,
            f" STRATEGY COMPARISON EXPERIMENT: {model_name} ({dataset})",
            "=" * 99,
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

                    d_ml = f"{((ml_val - max_length) / max_length * 100):+.1f}%" if ml_val else ""
                    d_wml = f"{((wml_val - max_length) / max_length * 100):+.1f}%" if wml_val else ""
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
        summary_text = "\n".join(lines)
        print(summary_text)

        # Save reports
        report_dir = os.path.join(repo_root, model_dir, "results", dataset)
        os.makedirs(report_dir, exist_ok=True)
        report_txt = os.path.join(report_dir, "strategy_comparison.txt")
        with open(report_txt, "w") as f:
            f.write(summary_text + "\n")

        report_json = os.path.join(report_dir, "strategy_comparison.json")
        with open(report_json, "w") as f:
            json.dump(table_data, f, indent=2)

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
        report_md = os.path.join(report_dir, "strategy_comparison.md")
        with open(report_md, "w") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"Saved text report to:     {report_txt}")
        print(f"Saved JSON report to:     {report_json}")
        print(f"Saved Markdown report to: {report_md}\n")


if __name__ == "__main__":
    main()

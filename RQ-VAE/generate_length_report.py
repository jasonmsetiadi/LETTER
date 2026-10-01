#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate length ablation report and plot across codebook depths (L=2, 4, 6, 8, 10)."
    )
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="rqvae",
        help="Tokenizer type: rqvae or letter (default: rqvae).",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Space- or comma-separated list of lengths to report (e.g. '2 4 6 8 10'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Varlen strategy to report (e.g. shortest_unique, popularity:cf_density). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tiger",
        help="Comma-separated list of models: tiger, lcrec (default: tiger).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate a comparison plot image.",
    )
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
    return parser.parse_args()


def autodetect_lengths(report_dirs, dataset):
    found_lengths = set()
    for d in report_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                if fname == "fixed.json" or (fname.startswith("varlen") and "_max" not in fname and "_min" not in fname):
                    found_lengths.add(4)
                m_fixed = re.match(r"fixed_L(\d+)\.json", fname)
                if m_fixed:
                    found_lengths.add(int(m_fixed.group(1)))
                m_varlen = re.search(r"varlen.*_max(\d+)\.json", fname)
                if m_varlen:
                    found_lengths.add(int(m_varlen.group(1)))
    return sorted(list(found_lengths)) if found_lengths else [2, 4, 6, 8, 10]


def generate_length_plot(table_data, dataset, model_name, tok_label, tested_lengths, out_png_path):
    """Generate a publication-quality 4-panel plot comparing fixed vs varlen across lengths."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend for remote server/headless environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Tip] matplotlib is not installed. To generate plot images, run: pip install matplotlib")
        return False

    fixed_data = {row["max_length"]: row for row in table_data if row["mode"] == "fixed"}
    varlen_data = {row["max_length"]: row for row in table_data if row["mode"] == "varlen"}

    x_lengths = sorted(tested_lengths)
    if not x_lengths:
        return False

    def extract_series(data_dict, metric_name):
        vals = []
        for l in x_lengths:
            row = data_dict.get(l, {})
            m = row.get("metrics", {})
            v = m.get(metric_name)
            vals.append(v * 100.0 if v is not None else None)
        return vals

    def extract_mean_len(data_dict):
        vals = []
        for l in x_lengths:
            row = data_dict.get(l, {})
            ml = row.get("mean_length", "")
            try:
                vals.append(float(str(ml).replace("~", "")))
            except Exception:
                vals.append(float(l))
        return vals

    fixed_h10 = extract_series(fixed_data, "hit@10")
    varlen_h10 = extract_series(varlen_data, "hit@10")
    fixed_n10 = extract_series(fixed_data, "ndcg@10")
    varlen_n10 = extract_series(varlen_data, "ndcg@10")
    fixed_h5 = extract_series(fixed_data, "hit@5")
    varlen_h5 = extract_series(varlen_data, "hit@5")
    fixed_len = extract_mean_len(fixed_data)
    varlen_len = extract_mean_len(varlen_data)

    # Check if there is valid data to plot
    if all(v is None for v in fixed_h10 + varlen_h10):
        return False

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    fig.suptitle(f"{tok_label} Codebook Depth Comparison: {model_name} ({dataset})", fontsize=14, fontweight="bold", y=0.98)

    def plot_metric(ax, fixed_series, varlen_series, ylabel, title):
        valid_fx = [(x, y) for x, y in zip(x_lengths, fixed_series) if y is not None]
        valid_vx = [(x, y) for x, y in zip(x_lengths, varlen_series) if y is not None]
        if valid_fx:
            ax.plot([p[0] for p in valid_fx], [p[1] for p in valid_fx], marker='o', linewidth=2.2, markersize=6, color='#1f77b4', label='Fixed-Length')
        if valid_vx:
            ax.plot([p[0] for p in valid_vx], [p[1] for p in valid_vx], marker='s', linewidth=2.2, markersize=6, linestyle='--', color='#ff7f0e', label='Variable-Length')
        ax.set_title(title, fontsize=12, fontweight='semibold')
        ax.set_xlabel("Max Codebook Depth (Tokens)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_lengths)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    # (a) Hit@10
    plot_metric(axes[0, 0], fixed_h10, varlen_h10, "Hit@10 (%)", "Hit@10 vs. Codebook Depth")

    # (b) NDCG@10
    plot_metric(axes[0, 1], fixed_n10, varlen_n10, "NDCG@10 (%)", "NDCG@10 vs. Codebook Depth")

    # (c) Hit@5
    plot_metric(axes[1, 0], fixed_h5, varlen_h5, "Hit@5 (%)", "Hit@5 vs. Codebook Depth")

    # (d) Sequence Length / Compression
    ax_len = axes[1, 1]
    ax_len.plot(x_lengths, fixed_len, marker='o', linewidth=2.2, markersize=6, color='#1f77b4', label='Fixed (L=Depth)')
    ax_len.plot(x_lengths, varlen_len, marker='s', linewidth=2.2, markersize=6, linestyle='--', color='#2ca02c', label='Varlen Effective Mean')
    ax_len.set_title("Decoding Sequence Length Compression", fontsize=12, fontweight='semibold')
    ax_len.set_xlabel("Max Codebook Depth (Tokens)", fontsize=10)
    ax_len.set_ylabel("Mean Token Sequence Length", fontsize=10)
    ax_len.set_xticks(x_lengths)
    ax_len.grid(True, linestyle="--", alpha=0.5)
    ax_len.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot image saved to:    {out_png_path}")
    return True


def main():
    args = parse_args()
    dataset = args.dataset
    tokenizer = (args.tokenizer or "rqvae").strip().lower()
    repo_root = args.repo_root
    data_root = args.data_root or os.path.join(repo_root, "data")
    models = [m.strip().lower() for m in args.models.replace(",", " ").split() if m.strip()]

    tok_label = "Vanilla RQ-VAE" if tokenizer == "rqvae" else "LETTER"

    for model in models:
        model_name = "TIGER" if model == "tiger" else "LC-Rec"
        model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"

        tok_report_dir = os.path.join(repo_root, model_dir, "results", dataset, tokenizer)
        legacy_report_dir = os.path.join(repo_root, model_dir, "results", dataset)
        search_dirs = [tok_report_dir, legacy_report_dir] if tokenizer == "letter" else [tok_report_dir]

        if args.lengths:
            tested_lengths = sorted(list(set(int(x) for x in args.lengths.replace(",", " ").split() if x.strip())))
        else:
            tested_lengths = autodetect_lengths(search_dirs, dataset)

        out_dir = tok_report_dir if (os.path.isdir(tok_report_dir) or tokenizer != "letter") else legacy_report_dir
        os.makedirs(out_dir, exist_ok=True)

        row_fmt = "| {:<8} | {:<8} | {:<6} | {:<7} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
        sep = "+----------+----------+--------+---------+----------+----------+----------+----------+----------+"
        header = row_fmt.format("Model", "Mode", "Max L", "Mean L", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10")

        rows = [
            "\n" + "=" * 90,
            f" {tok_label} Length Experiment Results Summary ({dataset} - {model_name})",
            "=" * 90,
            header,
            sep,
        ]

        table_data = []

        # Strategy resolution
        strat_tag = ""
        strat_suffix = ""
        if args.strategy:
            s_clean = args.strategy.strip().lower()
            strat_map = {
                "shortest_unique": ("", ""),
                "su": ("", ""),
                "popularity": ("-pop", ".pop"),
                "popularity:frequency": ("-pop", ".pop"),
                "popularity:user_entropy": ("-pop-entropy", ".pop-entropy"),
                "popularity:pagerank": ("-pop-pagerank", ".pop-pagerank"),
                "popularity:co_occurrence": ("-pop-cooccur", ".pop-cooccur"),
                "popularity:cf_density": ("-pop-cf", ".pop-cf"),
                "residual": ("-res", ".res"),
            }
            strat_tag, strat_suffix = strat_map.get(s_clean, (f"-{s_clean}", f".{s_clean}"))

        for l in tested_lengths:
            for mode in ["fixed", "varlen"]:
                fpath = None
                if mode == "fixed":
                    fname = "fixed.json" if l == 4 else f"fixed_L{l}.json"
                    mean_l = f"{l}.00"
                    for sdir in search_dirs:
                        p = os.path.join(sdir, fname)
                        if os.path.isfile(p):
                            fpath = p
                            break
                else:
                    cand_fnames = []
                    if strat_tag:
                        cand_fnames.append(f"varlen{strat_tag}.json" if l == 4 else f"varlen{strat_tag}_max{l}.json")
                    cand_fnames.append("varlen.json" if l == 4 else f"varlen_max{l}.json")

                    for fn in cand_fnames:
                        for sdir in search_dirs:
                            p = os.path.join(sdir, fn)
                            if os.path.isfile(p):
                                fpath = p
                                break
                        if fpath:
                            break

                    if not fpath:
                        for sdir in search_dirs:
                            if os.path.isdir(sdir):
                                matches = glob.glob(os.path.join(sdir, f"varlen*{l}.json"))
                                if matches:
                                    fpath = sorted(matches)[0]
                                    break

                    cand_summary_names = []
                    if strat_suffix:
                        cand_summary_names.append(f"{dataset}.index.varlen{strat_suffix}.summary.json" if l == 4 else f"{dataset}.index.varlen{strat_suffix}.max{l}.summary.json")
                    cand_summary_names.append(f"{dataset}.index.varlen.summary.json" if l == 4 else f"{dataset}.index.varlen.max{l}.summary.json")

                    summary_path = None
                    for sname in cand_summary_names:
                        for sdir in [os.path.join(data_root, dataset, tokenizer), os.path.join(data_root, dataset)]:
                            p = os.path.join(sdir, sname)
                            if os.path.isfile(p):
                                summary_path = p
                                break
                        if summary_path:
                            break

                    if summary_path:
                        try:
                            with open(summary_path) as sf:
                                sdata = json.load(sf)
                            ml_val = sdata.get("mean_length", float(l))
                            mean_l = f"{ml_val:.2f}"
                        except Exception:
                            mean_l = f"~{l}"
                    else:
                        mean_l = f"~{l}"

                entry = {
                    "model": model_name,
                    "mode": mode,
                    "max_length": l,
                    "mean_length": mean_l,
                    "metrics": {},
                    "status": "pending",
                }

                if fpath and os.path.isfile(fpath):
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        res = data.get("mean_results", {})
                        entry["metrics"] = res
                        entry["status"] = "completed"
                        v1 = res.get("hit@1")
                        v5 = res.get("hit@5")
                        v10 = res.get("hit@10")
                        vn5 = res.get("ndcg@5")
                        vn10 = res.get("ndcg@10")
                        h1 = f"{v1 * 100:.2f}%" if v1 is not None else "-"
                        h5 = f"{v5 * 100:.2f}%" if v5 is not None else "-"
                        h10 = f"{v10 * 100:.2f}%" if v10 is not None else "-"
                        n5 = f"{vn5 * 100:.2f}%" if vn5 is not None else "-"
                        n10 = f"{vn10 * 100:.2f}%" if vn10 is not None else "-"
                        rows.append(row_fmt.format(model_name, mode, str(l), mean_l, h1, h5, h10, n5, n10))
                    except Exception:
                        rows.append(row_fmt.format(model_name, mode, str(l), mean_l, "(err)", "-", "-", "-", "-"))
                        entry["status"] = "error"
                else:
                    rows.append(row_fmt.format(model_name, mode, str(l), mean_l, "(pending)", "-", "-", "-", "-"))

                table_data.append(entry)

        rows.append(sep)
        summary_text = "\n".join(rows)
        print(summary_text)

        # Plot generation
        plot_saved = False
        if not args.no_plot:
            plot_png_path = os.path.join(out_dir, f"experiment_summary_{dataset}.png")
            plot_saved = generate_length_plot(
                table_data=table_data,
                dataset=dataset,
                model_name=model_name,
                tok_label=tok_label,
                tested_lengths=tested_lengths,
                out_png_path=plot_png_path,
            )

        # Markdown report
        md_lines = [
            f"# {tok_label} Codebook Length Experiment Summary: {model_name} ({dataset})\n",
            f"- **Dataset**: `{dataset}`",
            f"- **Model**: `{model_name}`",
            f"- **Tokenizer**: `{tokenizer}`",
            f"- **Evaluated Depths**: {', '.join(str(x) for x in tested_lengths)}\n",
        ]
        if plot_saved:
            md_lines.extend([
                "### Performance Curves\n",
                f"![{tok_label} Length Comparison Plot](experiment_summary_{dataset}.png)\n",
            ])
        md_lines.extend([
            "### 1. Recommendation Performance by Codebook Depth\n",
            "| Model | Mode | Max L | Mean L | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in table_data:
            m = row["metrics"]
            v1 = f"{m['hit@1'] * 100:.2f}%" if m.get("hit@1") is not None else "-"
            v5 = f"{m['hit@5'] * 100:.2f}%" if m.get("hit@5") is not None else "-"
            v10 = f"{m['hit@10'] * 100:.2f}%" if m.get("hit@10") is not None else "-"
            vn5 = f"{m['ndcg@5'] * 100:.2f}%" if m.get("ndcg@5") is not None else "-"
            vn10 = f"{m['ndcg@10'] * 100:.2f}%" if m.get("ndcg@10") is not None else "-"
            md_lines.append(
                f"| **{row['model']}** | {row['mode']} | {row['max_length']} | {row['mean_length']} | "
                f"{v1} | {v5} | {v10} | {vn5} | {vn10} | {row['status']} |"
            )
        md_lines.append("")

        summary_md_path = os.path.join(out_dir, f"experiment_summary_{dataset}.md")
        with open(summary_md_path, "w") as mf:
            mf.write("\n".join(md_lines))
        print(f"Markdown table saved to: {summary_md_path}\n")


if __name__ == "__main__":
    main()

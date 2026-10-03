#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


TOKENIZER_LABELS = {
    "rqvae": "Vanilla RQ-VAE",
    "letter": "LETTER",
}


def get_tokenizer_label(tokenizer_name):
    """Return a human-friendly label for a tokenizer."""
    tok_clean = tokenizer_name.strip()
    return TOKENIZER_LABELS.get(tok_clean.lower(), tok_clean.upper())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed-length codebook depth ablation report and comparison across tokenizers."
    )
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name.")
    parser.add_argument(
        "--tokenizers",
        "--tokenizer",
        dest="tokenizers",
        type=str,
        default="letter,rqvae",
        help="Comma- or space-separated list of tokenizers to compare (e.g. 'letter,rqvae' or 'auto'). Default: 'letter,rqvae'.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Space- or comma-separated list of fixed lengths to report (e.g. '2 4 6 8 10'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tiger",
        help="Comma-separated list of models: tiger, lcrec (default: tiger).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory to save report and plot. Defaults to results/<dataset>/ (multi-tokenizer) or results/<dataset>/<tokenizer>/ (single tokenizer).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate a comparison plot image.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Repository root directory.",
    )
    return parser.parse_args()


def resolve_tokenizers(tok_arg, search_base_dir):
    """Resolve tokenizers list, auto-discovering if 'auto' or 'all' is passed."""
    if not tok_arg or tok_arg.strip().lower() in ("auto", "all"):
        found = []
        if os.path.isdir(search_base_dir):
            for entry in sorted(os.listdir(search_base_dir)):
                p = os.path.join(search_base_dir, entry)
                if os.path.isdir(p) and glob.glob(os.path.join(p, "fixed*.json")):
                    found.append(entry)
        if found:
            return found
        return ["letter", "rqvae"]
    return [t.strip().lower() for t in tok_arg.replace(",", " ").split() if t.strip()]


def autodetect_lengths(tokenizers, search_base_dir):
    """Find all evaluated fixed codebook depths across specified tokenizers."""
    found_lengths = set()
    for tok in tokenizers:
        tok_dir = os.path.join(search_base_dir, tok)
        if not os.path.isdir(tok_dir):
            continue
        for fname in os.listdir(tok_dir):
            if not fname.endswith(".json"):
                continue
            m = re.match(r"^fixed_L(\d+)\.json$", fname)
            if m:
                found_lengths.add(int(m.group(1)))
    return sorted(list(found_lengths)) if found_lengths else [2, 4, 6, 8, 10]


def find_fixed_result_file(tok_dir, length):
    """Locate fixed-length result json for a given codebook depth."""
    if not os.path.isdir(tok_dir):
        return None
    p = os.path.join(tok_dir, f"fixed_L{length}.json")
    return p if os.path.isfile(p) else None


def generate_length_plot(table_data, dataset, model_name, tokenizers, tested_lengths, out_png_path):
    """Generate a 4-panel publication-quality plot comparing tokenizers across fixed lengths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Tip] matplotlib is not installed. To generate plot images, run: pip install matplotlib")
        return False

    x_lengths = sorted(tested_lengths)
    if not x_lengths:
        return False

    # Check if any tokenizer has at least one valid metric
    has_any_data = any(
        any(row.get("metrics", {}).get(m) is not None for m in ["hit@10", "hit@5", "ndcg@10", "ndcg@5"])
        for row in table_data
    )
    if not has_any_data:
        return False

    data_by_tok = {}
    for row in table_data:
        tok = row["tokenizer"]
        if tok not in data_by_tok:
            data_by_tok[tok] = {}
        data_by_tok[tok][row["length"]] = row.get("metrics", {})

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    markers = ["o", "s", "^", "D", "v", "p", "P", "*", "X", "<"]
    linestyles = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    fig.suptitle(
        f"Fixed-Length Codebook Depth Comparison: {model_name} ({dataset})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    metrics_to_plot = [
        ("hit@10", "Hit@10 (%)", "Hit@10 vs. Codebook Depth", axes[0, 0]),
        ("ndcg@10", "NDCG@10 (%)", "NDCG@10 vs. Codebook Depth", axes[0, 1]),
        ("hit@5", "Hit@5 (%)", "Hit@5 vs. Codebook Depth", axes[1, 0]),
        ("ndcg@5", "NDCG@5 (%)", "NDCG@5 vs. Codebook Depth", axes[1, 1]),
    ]

    global_legend_items = {}

    for metric_key, ylabel, title, ax in metrics_to_plot:
        for idx, tok in enumerate(tokenizers):
            label = get_tokenizer_label(tok)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            linestyle = linestyles[(idx // len(colors)) % len(linestyles)]

            tok_metrics = data_by_tok.get(tok, {})
            valid_pts = []
            for l in x_lengths:
                val = tok_metrics.get(l, {}).get(metric_key)
                if val is not None:
                    valid_pts.append((l, val * 100.0))

            if valid_pts:
                ax.plot(
                    [p[0] for p in valid_pts],
                    [p[1] for p in valid_pts],
                    marker=marker,
                    linewidth=2.2,
                    markersize=6,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )

        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Codebook Depth / Sequence Length (L)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_lengths)
        ax.grid(True, linestyle="--", alpha=0.5)

        try:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l and l not in global_legend_items:
                    global_legend_items[l] = h
        except (TypeError, ValueError):
            pass

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    if global_legend_items:
        fig.legend(
            global_legend_items.values(),
            global_legend_items.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(global_legend_items)),
            fontsize=9.5,
            frameon=True,
            framealpha=0.95,
        )

    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot image saved to:    {out_png_path}")
    return True


def main():
    args = parse_args()
    dataset = args.dataset
    repo_root = args.repo_root
    models = [m.strip().lower() for m in args.models.replace(",", " ").split() if m.strip()]

    for model in models:
        model_name = "TIGER" if model == "tiger" else "LC-Rec"
        model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"
        results_dataset_dir = os.path.join(repo_root, model_dir, "results", dataset)

        tokenizers = resolve_tokenizers(args.tokenizers, results_dataset_dir)

        if args.lengths:
            tested_lengths = sorted(list(set(int(x) for x in args.lengths.replace(",", " ").split() if x.strip())))
        else:
            tested_lengths = autodetect_lengths(tokenizers, results_dataset_dir)

        # Output directory determination: always save at dataset root (results/<dataset>/)
        out_dir = args.output_dir if args.output_dir else results_dataset_dir
        os.makedirs(out_dir, exist_ok=True)

        row_fmt = "| {:<16} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10} |"
        sep = "+------------------+----------+----------+----------+----------+----------+----------+------------+"
        header = row_fmt.format("Tokenizer", "Length", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10", "Status")

        rows = [
            "\n" + "=" * 94,
            f" Fixed-Length Codebook Depth Evaluation Summary ({dataset} - {model_name})",
            "=" * 94,
            header,
            sep,
        ]

        table_data = []

        # Iterate by length then tokenizer to facilitate comparison
        for l in tested_lengths:
            for tok in tokenizers:
                tok_label = get_tokenizer_label(tok)
                tok_dir = os.path.join(results_dataset_dir, tok)
                fpath = find_fixed_result_file(tok_dir, l)

                entry = {
                    "model": model_name,
                    "tokenizer": tok,
                    "tokenizer_label": tok_label,
                    "length": l,
                    "metrics": {},
                    "status": "pending",
                }

                if fpath and os.path.isfile(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
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
                        rows.append(row_fmt.format(tok_label, str(l), h1, h5, h10, n5, n10, "completed"))
                    except Exception:
                        rows.append(row_fmt.format(tok_label, str(l), "(err)", "-", "-", "-", "-", "error"))
                        entry["status"] = "error"
                else:
                    rows.append(row_fmt.format(tok_label, str(l), "-", "-", "-", "-", "-", "pending"))

                table_data.append(entry)

        rows.append(sep)
        summary_text = "\n".join(rows)
        print(summary_text)

        # Plot generation
        plot_saved = False
        plot_fname = f"fixed_length_comparison_{dataset}.png"
        plot_png_path = os.path.join(out_dir, plot_fname)
        if not args.no_plot:
            plot_saved = generate_length_plot(
                table_data=table_data,
                dataset=dataset,
                model_name=model_name,
                tokenizers=tokenizers,
                tested_lengths=tested_lengths,
                out_png_path=plot_png_path,
            )

        # Markdown report
        tok_display_list = ", ".join(f"`{get_tokenizer_label(t)}`" for t in tokenizers)
        md_lines = [
            f"# Fixed-Length Codebook Depth Comparison: {model_name} ({dataset})\n",
            f"- **Dataset**: `{dataset}`",
            f"- **Model**: `{model_name}`",
            f"- **Tokenizers Evaluated**: {tok_display_list}",
            f"- **Evaluated Depths (L)**: {', '.join(str(x) for x in tested_lengths)}\n",
        ]

        if plot_saved:
            md_lines.extend([
                "### Performance Curves\n",
                f"![Fixed-Length Comparison Plot]({plot_fname})\n",
            ])

        md_lines.extend([
            "### 1. Recommendation Performance Across Codebook Depths\n",
            "| Tokenizer | Length (L) | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for row in table_data:
            m = row["metrics"]
            v1 = f"{m['hit@1'] * 100:.2f}%" if m.get("hit@1") is not None else "-"
            v5 = f"{m['hit@5'] * 100:.2f}%" if m.get("hit@5") is not None else "-"
            v10 = f"{m['hit@10'] * 100:.2f}%" if m.get("hit@10") is not None else "-"
            vn5 = f"{m['ndcg@5'] * 100:.2f}%" if m.get("ndcg@5") is not None else "-"
            vn10 = f"{m['ndcg@10'] * 100:.2f}%" if m.get("ndcg@10") is not None else "-"
            md_lines.append(
                f"| **{row['tokenizer_label']}** | {row['length']} | "
                f"{v1} | {v5} | {v10} | {vn5} | {vn10} | {row['status']} |"
            )
        md_lines.append("")

        # Side-by-side comparison tables if multiple tokenizers
        if len(tokenizers) > 1:
            tok_labels = [get_tokenizer_label(t) for t in tokenizers]

            # Hit@10 table
            h10_header = "| Length (L) | " + " | ".join(tok_labels) + " | Best |"
            h10_sep = "| :---: | " + " | ".join([":---:" for _ in tok_labels]) + " | :---: |"
            md_lines.extend([
                "### 2. Hit@10 (%) Comparison Across Tokenizers\n",
                h10_header,
                h10_sep,
            ])
            for l in tested_lengths:
                vals = []
                best_label = "-"
                best_v = -1.0
                for tok in tokenizers:
                    matched = [r for r in table_data if r["length"] == l and r["tokenizer"] == tok]
                    val = matched[0]["metrics"].get("hit@10") if matched else None
                    if val is not None:
                        vals.append(f"{val * 100:.2f}%")
                        if val > best_v:
                            best_v = val
                            best_label = get_tokenizer_label(tok)
                    else:
                        vals.append("-")
                md_lines.append(f"| {l} | " + " | ".join(vals) + f" | **{best_label}** |")
            md_lines.append("")

            # NDCG@10 table
            n10_header = "| Length (L) | " + " | ".join(tok_labels) + " | Best |"
            n10_sep = "| :---: | " + " | ".join([":---:" for _ in tok_labels]) + " | :---: |"
            md_lines.extend([
                "### 3. NDCG@10 (%) Comparison Across Tokenizers\n",
                n10_header,
                n10_sep,
            ])
            for l in tested_lengths:
                vals = []
                best_label = "-"
                best_v = -1.0
                for tok in tokenizers:
                    matched = [r for r in table_data if r["length"] == l and r["tokenizer"] == tok]
                    val = matched[0]["metrics"].get("ndcg@10") if matched else None
                    if val is not None:
                        vals.append(f"{val * 100:.2f}%")
                        if val > best_v:
                            best_v = val
                            best_label = get_tokenizer_label(tok)
                    else:
                        vals.append("-")
                md_lines.append(f"| {l} | " + " | ".join(vals) + f" | **{best_label}** |")
            md_lines.append("")

        report_fname = f"fixed_length_comparison_{dataset}.md"
        summary_md_path = os.path.join(out_dir, report_fname)
        with open(summary_md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines) + "\n")
        print(f"Markdown table saved to: {summary_md_path}\n")


if __name__ == "__main__":
    main()

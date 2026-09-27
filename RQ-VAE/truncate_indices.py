"""Create a collision-free variable-length LETTER index from fixed-length IDs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_frequencies(inter_file_path):
    """Load item interaction frequencies from a .inter.json file or frequency dict."""
    with Path(inter_file_path).open(encoding="utf-8") as f:
        data = json.load(f)
    freqs = defaultdict(int)
    for k, v in data.items():
        if isinstance(v, list):
            for item in v:
                freqs[str(item)] += 1
        elif isinstance(v, (int, float)):
            freqs[str(k)] = int(v)
    return freqs


def load_residuals(residuals_file_path):
    """Load layer-wise reconstruction residuals from a JSON file."""
    with Path(residuals_file_path).open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def truncate_indices(
    indices,
    min_length=1,
    max_length=4,
    allow_base_collisions=True,
    strategy="shortest_unique",
    item_frequencies=None,
    residuals=None,
    residual_threshold=0.2,
):
    """Truncate each item according to the selected variable-length strategy.

    Strategies:
      - 'shortest_unique': Truncate to the shortest catalog-unique prefix in [min_length, max_length].
      - 'popularity': Partition items into length tiers based on empirical frequency distribution.
                      High-frequency items can truncate down to min_length; rare tail items
                      are preserved at max_length.
      - 'residual': Truncate to the shortest unique prefix whose cumulative reconstruction error
                    is at or below residual_threshold.
    """
    valid_strategies = {"shortest_unique", "popularity", "residual"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid strategies are: {sorted(valid_strategies)}"
        )

    if not 1 <= min_length <= max_length:
        raise ValueError("Lengths must satisfy 1 <= min_length <= max_length.")

    for item_id, tokens in indices.items():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Item {item_id} must have a non-empty token list.")
        if len(tokens) < max_length:
            raise ValueError(
                f"Item {item_id} has {len(tokens)} tokens, fewer than max_length={max_length}."
            )

    prefix_counts = {l: defaultdict(int) for l in range(1, max_length + 1)}
    for tokens in indices.values():
        for l in range(1, max_length + 1):
            prefix_counts[l][tuple(tokens[:l])] += 1

    item_min_len = {}
    if strategy == "popularity":
        if item_frequencies is None:
            raise ValueError("strategy='popularity' requires item_frequencies.")
        k_levels = max_length - min_length + 1
        if k_levels <= 1:
            for item_id in indices:
                item_min_len[item_id] = max_length
        else:
            sorted_items = sorted(
                indices.keys(), key=lambda k: (item_frequencies.get(str(k), 0), str(k))
            )
            n_items = len(sorted_items)
            for rank, item_id in enumerate(sorted_items):
                tier = min(int(rank * k_levels / n_items), k_levels - 1)
                item_min_len[item_id] = max_length - tier
    elif strategy == "residual":
        if residuals is None:
            raise ValueError("strategy='residual' requires residuals.")

    truncated = {}
    lengths = {}

    for item_id, tokens in indices.items():
        chosen_len = max_length

        if strategy == "shortest_unique":
            for l in range(min_length, max_length):
                if prefix_counts[l][tuple(tokens[:l])] == 1:
                    chosen_len = l
                    break

        elif strategy == "popularity":
            start_l = item_min_len[item_id]
            for l in range(start_l, max_length):
                if prefix_counts[l][tuple(tokens[:l])] == 1:
                    chosen_len = l
                    break

        elif strategy == "residual":
            item_res = residuals.get(str(item_id))
            for l in range(min_length, max_length):
                if prefix_counts[l][tuple(tokens[:l])] == 1:
                    err = 1.0
                    if isinstance(item_res, list) and len(item_res) >= l:
                        err = item_res[l - 1]
                    elif isinstance(item_res, dict):
                        err = item_res.get(str(l), item_res.get(l, 1.0))

                    if err <= residual_threshold:
                        chosen_len = l
                        break

        if chosen_len == max_length and not allow_base_collisions:
            if prefix_counts[max_length][tuple(tokens[:max_length])] > 1:
                raise ValueError(
                    "The selected maximum length cannot produce unique semantic IDs."
                )

        truncated[item_id] = tokens[:chosen_len]
        lengths[item_id] = chosen_len

    return truncated, lengths


def main():
    parser = argparse.ArgumentParser(
        description="Truncate fixed-length LETTER IDs into collision-free variable-length IDs."
    )
    parser.add_argument("--input", required=True, help="Input .index.json file.")
    parser.add_argument("--output", required=True, help="Output variable-length .index.json file.")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=["shortest_unique", "popularity", "residual"],
        default="shortest_unique",
        help="Variable-length truncation strategy (default: shortest_unique).",
    )
    parser.add_argument(
        "--inter-file",
        type=str,
        default=None,
        help="Path to interaction JSON file (<dataset>.inter.json or {item: count}) for popularity strategy.",
    )
    parser.add_argument(
        "--residuals-file",
        type=str,
        default=None,
        help="Path to layer-wise residuals JSON file for residual strategy.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.2,
        help="Maximum allowable reconstruction residual error for residual strategy (default: 0.2).",
    )
    parser.add_argument(
        "--allow-base-collisions",
        action="store_true",
        default=True,
        help="Allow inherent collisions that already exist in the input index at max_length (default: True).",
    )
    parser.add_argument(
        "--strict",
        action="store_false",
        dest="allow_base_collisions",
        help="Disallow any collisions and raise an error if unique IDs cannot be produced.",
    )
    args = parser.parse_args()

    if args.strategy == "popularity" and not args.inter_file:
        parser.error("--strategy popularity requires --inter-file.")
    if args.strategy == "residual" and not args.residuals_file:
        parser.error("--strategy residual requires --residuals-file.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open(encoding="utf-8") as input_file:
        indices = json.load(input_file)

    item_frequencies = None
    if args.inter_file:
        item_frequencies = load_frequencies(args.inter_file)

    residuals = None
    if args.residuals_file:
        residuals = load_residuals(args.residuals_file)

    truncated, lengths = truncate_indices(
        indices,
        min_length=args.min_length,
        max_length=args.max_length,
        allow_base_collisions=args.allow_base_collisions,
        strategy=args.strategy,
        item_frequencies=item_frequencies,
        residuals=residuals,
        residual_threshold=args.residual_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(truncated, output_file)

    unique_truncated = len({tuple(v) for v in truncated.values()})
    unique_base = len({tuple(v[:args.max_length]) for v in indices.values()})
    summary_path = output_path.with_suffix(".summary.json")

    from collections import Counter
    length_distribution = dict(sorted(Counter(lengths.values()).items()))

    summary_payload = {
        "input": str(input_path),
        "strategy": args.strategy,
        "items": len(truncated),
        "min_length": min(lengths.values()),
        "max_length": max(lengths.values()),
        "mean_length": sum(lengths.values()) / len(lengths),
        "length_distribution": length_distribution,
        "unique_ids": unique_truncated,
        "collisions": len(truncated) - unique_truncated,
        "base_collisions": len(indices) - unique_base,
    }

    if args.strategy == "popularity":
        summary_payload["inter_file"] = str(args.inter_file)
        if item_frequencies:
            weighted_length = sum(
                lengths[item_id] * item_frequencies.get(str(item_id), 0)
                for item_id in truncated
            ) / max(1, sum(item_frequencies.get(str(item_id), 0) for item_id in truncated))
            summary_payload["weighted_mean_length"] = weighted_length
    elif args.strategy == "residual":
        summary_payload["residuals_file"] = str(args.residuals_file)
        summary_payload["residual_threshold"] = args.residual_threshold

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            summary_payload,
            summary_file,
            indent=2,
        )

    print(f"Saved {len(truncated)} variable-length IDs ({args.strategy}) to {output_path}")
    print(f"Saved length summary to {summary_path}")


if __name__ == "__main__":
    main()

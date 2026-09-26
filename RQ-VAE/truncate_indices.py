"""Create a collision-free variable-length LETTER index from fixed-length IDs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def truncate_indices(indices, min_length=1, max_length=4, allow_base_collisions=True):
    """Truncate each item to its shortest prefix that is unique across the catalog."""
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

    truncated = {}
    lengths = {}

    for item_id, tokens in indices.items():
        chosen_len = max_length
        for l in range(min_length, max_length):
            if prefix_counts[l][tuple(tokens[:l])] == 1:
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

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open(encoding="utf-8") as input_file:
        indices = json.load(input_file)

    truncated, lengths = truncate_indices(
        indices,
        min_length=args.min_length,
        max_length=args.max_length,
        allow_base_collisions=args.allow_base_collisions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(truncated, output_file)

    unique_truncated = len({tuple(v) for v in truncated.values()})
    unique_base = len({tuple(v[:args.max_length]) for v in indices.values()})
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "input": str(input_path),
                "items": len(truncated),
                "min_length": min(lengths.values()),
                "max_length": max(lengths.values()),
                "mean_length": sum(lengths.values()) / len(lengths),
                "unique_ids": unique_truncated,
                "collisions": len(truncated) - unique_truncated,
                "base_collisions": len(indices) - unique_base,
            },
            summary_file,
            indent=2,
        )

    print(f"Saved {len(truncated)} variable-length IDs to {output_path}")
    print(f"Saved length summary to {summary_path}")


if __name__ == "__main__":
    main()

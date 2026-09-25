"""Create a collision-free variable-length LETTER index from fixed-length IDs."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def truncate_indices(indices, min_length, max_length, seed):
    if not 1 <= min_length <= max_length:
        raise ValueError("Lengths must satisfy 1 <= min_length <= max_length.")

    rng = random.Random(seed)
    truncated = {}
    lengths = {}

    for item_id, tokens in indices.items():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Item {item_id} must have a non-empty token list.")
        if len(tokens) < max_length:
            raise ValueError(
                f"Item {item_id} has {len(tokens)} tokens, fewer than max_length={max_length}."
            )
        length = rng.randint(min_length, max_length)
        truncated[item_id] = tokens[:length]
        lengths[item_id] = length

    while True:
        collisions = defaultdict(list)
        for item_id, tokens in truncated.items():
            collisions[tuple(tokens)].append(item_id)

        duplicate_groups = [group for group in collisions.values() if len(group) > 1]
        if not duplicate_groups:
            return truncated, lengths

        changed = False
        for group in duplicate_groups:
            for item_id in group:
                if lengths[item_id] < max_length:
                    lengths[item_id] += 1
                    truncated[item_id] = indices[item_id][:lengths[item_id]]
                    changed = True
        if not changed:
            raise ValueError(
                "The selected maximum length cannot produce unique semantic IDs."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Randomly truncate fixed-length LETTER IDs into variable-length IDs."
    )
    parser.add_argument("--input", required=True, help="Input .index.json file.")
    parser.add_argument("--output", required=True, help="Output variable-length .index.json file.")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open(encoding="utf-8") as input_file:
        indices = json.load(input_file)

    truncated, lengths = truncate_indices(
        indices, args.min_length, args.max_length, args.seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(truncated, output_file)

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "input": str(input_path),
                "seed": args.seed,
                "items": len(truncated),
                "min_length": min(lengths.values()),
                "max_length": max(lengths.values()),
                "mean_length": sum(lengths.values()) / len(lengths),
            },
            summary_file,
            indent=2,
        )

    print(f"Saved {len(truncated)} variable-length IDs to {output_path}")
    print(f"Saved length summary to {summary_path}")


if __name__ == "__main__":
    main()

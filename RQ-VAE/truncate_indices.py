"""Create a collision-free variable-length LETTER index from fixed-length IDs."""

import argparse
import json
import math
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


def compute_cf_density_scores(cf_emb_file, top_k=10):
    """Compute CF latent space isolation scores from embeddings.

    Higher score indicates that the item is more isolated (lower k-NN crowd density),
    making it safer to assign a shorter prefix without causing collaborative collisions.
    """
    cf_path = Path(cf_emb_file)
    if not cf_path.exists():
        raise FileNotFoundError(f"CF embedding file not found: {cf_path}")

    torch_mod = None
    try:
        import torch
        torch_mod = torch
    except ImportError:
        pass

    np_mod = None
    try:
        import numpy as np
        np_mod = np
    except ImportError:
        pass

    if cf_path.suffix == ".pt":
        if torch_mod is None:
            raise ImportError(
                "Loading .pt CF embeddings requires 'torch'. Please run in an environment with PyTorch installed."
            )
        try:
            tensor = torch_mod.load(cf_path, weights_only=False)
        except TypeError:
            tensor = torch_mod.load(cf_path)
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
        if hasattr(tensor, "dim") and tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        num_items = tensor.size(0)
        norm = tensor / torch_mod.clamp(tensor.norm(dim=1, keepdim=True), min=1e-8)
        k_val = min(top_k + 1, num_items)
        densities = []
        chunk_size = 2048
        for i in range(0, num_items, chunk_size):
            chunk = norm[i : i + chunk_size]
            sim = torch_mod.matmul(chunk, norm.t())
            topk_sim, _ = torch_mod.topk(sim, k=k_val, dim=1)
            if k_val > 1:
                avg_sim = topk_sim[:, 1:].mean(dim=1)
            else:
                avg_sim = topk_sim[:, 0]
            densities.extend(avg_sim.tolist())
        return {str(idx): (1.0 - float(d)) for idx, d in enumerate(densities)}

    if cf_path.suffix == ".npy":
        if np_mod is None:
            raise ImportError(
                "Loading .npy CF embeddings requires 'numpy'. Please run in an environment with NumPy installed."
            )
        arr = np_mod.load(cf_path)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        num_items = arr.shape[0]
        norm = arr / np_mod.clip(np_mod.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)
        k_val = min(top_k + 1, num_items)
        densities = []
        chunk_size = 2048
        for i in range(0, num_items, chunk_size):
            chunk = norm[i : i + chunk_size]
            sim = np_mod.dot(chunk, norm.T)
            partitioned = np_mod.partition(sim, -k_val, axis=1)[:, -k_val:]
            sorted_topk = np_mod.sort(partitioned, axis=1)[:, ::-1]
            if k_val > 1:
                avg_sim = sorted_topk[:, 1:].mean(axis=1)
            else:
                avg_sim = sorted_topk[:, 0]
            densities.extend(avg_sim.tolist())
        return {str(idx): (1.0 - float(d)) for idx, d in enumerate(densities)}

    if cf_path.suffix == ".json":
        with cf_path.open(encoding="utf-8") as f:
            data = json.load(f)
        scores = {}
        item_ids = list(data.keys())
        vectors = {}
        for it in item_ids:
            vec = data[it]
            mag = math.sqrt(sum(x * x for x in vec))
            vectors[it] = [x / mag for x in vec] if mag > 1e-8 else vec
        k_val = min(top_k, len(item_ids) - 1)
        for it in item_ids:
            v_it = vectors[it]
            sims = []
            for other in item_ids:
                if other == it:
                    continue
                v_ot = vectors[other]
                sims.append(sum(a * b for a, b in zip(v_it, v_ot)))
            sims.sort(reverse=True)
            avg_sim = sum(sims[:k_val]) / max(1, k_val) if k_val > 0 else 0.0
            scores[str(it)] = 1.0 - avg_sim
        return scores

    raise ValueError(
        f"Unsupported CF embedding format: {cf_path.suffix} (expected .pt, .npy, or .json)"
    )


def compute_interaction_signals(
    inter_source,
    signal="frequency",
    cf_emb_file=None,
    top_k_cf=10,
    damping=0.85,
    max_pagerank_iters=30,
):
    """Compute item priority scores from interaction sequences and/or CF embeddings.

    Supported signals:
      - 'frequency': Raw item interaction count (standard unigram popularity).
      - 'user_entropy': Shannon entropy over the user audience distribution.
                        Penalizes items whose interaction volume is driven by few power users.
      - 'target': Frequency of appearing in the sequence-final (target) position,
                  directly optimizing inference-time generation tokens.
      - 'pagerank': Stationary distribution on the sequential transition graph.
                    Identifies global transition hubs across recommendation paths.
      - 'co_occurrence': Degree/breadth of unique co-occurring catalog items across user sessions.
                        Identifies universal basket-companion items.
      - 'cf_density': Latent manifold isolation (1 - k-NN crowd density).
                      Requires cf_emb_file.
    """
    if isinstance(inter_source, (str, Path)):
        with Path(inter_source).open(encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(inter_source, dict):
        data = inter_source
    else:
        raise TypeError(f"inter_source must be a path or dict, got {type(inter_source)}")

    raw_freqs = defaultdict(int)
    user_counts = defaultdict(lambda: defaultdict(int))
    target_counts = defaultdict(int)
    transitions = defaultdict(lambda: defaultdict(int))
    out_degrees = defaultdict(int)
    cooccur_sets = defaultdict(set)
    has_user_sequences = False

    for k, v in data.items():
        if isinstance(v, list):
            has_user_sequences = True
            seq = v
            if len(seq) == 0:
                continue
            target_counts[str(seq[-1])] += 1
            unique_in_seq = set(str(item) for item in seq)
            for it in unique_in_seq:
                cooccur_sets[it].update(unique_in_seq - {it})
            for idx, item in enumerate(seq):
                item_str = str(item)
                raw_freqs[item_str] += 1
                user_counts[item_str][str(k)] += 1
                if idx < len(seq) - 1:
                    next_str = str(seq[idx + 1])
                    transitions[item_str][next_str] += 1
                    out_degrees[item_str] += 1
        elif isinstance(v, (int, float)):
            raw_freqs[str(k)] = int(v)

    canonical_signal = signal.lower().strip()

    if canonical_signal in ("frequency", "raw", "pop"):
        return dict(raw_freqs), dict(raw_freqs)

    if canonical_signal in ("user_entropy", "entropy"):
        if not has_user_sequences:
            return {k: math.log2(1.0 + v) for k, v in raw_freqs.items()}, dict(raw_freqs)
        entropy_scores = {}
        for item, u_dict in user_counts.items():
            tot = raw_freqs[item]
            if tot <= 0:
                entropy_scores[item] = 0.0
                continue
            h = 0.0
            for _, cnt in u_dict.items():
                p = cnt / tot
                if p > 0:
                    h -= p * math.log2(p)
            entropy_scores[item] = h
        return entropy_scores, dict(raw_freqs)

    if canonical_signal in ("target", "target_frequency"):
        if not has_user_sequences:
            return dict(raw_freqs), dict(raw_freqs)
        scores = {item: float(target_counts.get(item, 0)) for item in raw_freqs}
        return scores, dict(raw_freqs)

    if canonical_signal in ("pagerank", "pr"):
        if not has_user_sequences or not transitions:
            return dict(raw_freqs), dict(raw_freqs)
        items = list(raw_freqs.keys())
        n = len(items)
        if n == 0:
            return {}, dict(raw_freqs)
        pr = {it: 1.0 / n for it in items}
        for _ in range(max_pagerank_iters):
            dangling_sum = sum(pr[it] for it in items if out_degrees[it] == 0)
            base = (1.0 - damping + damping * dangling_sum) / n
            new_pr = {it: base for it in items}
            for src, targets in transitions.items():
                if out_degrees[src] == 0:
                    continue
                share = damping * pr[src] / out_degrees[src]
                for tgt, weight in targets.items():
                    new_pr[tgt] += share * weight
            diff = sum(abs(new_pr[it] - pr[it]) for it in items)
            pr = new_pr
            if diff < 1e-6:
                break
        return pr, dict(raw_freqs)

    if canonical_signal in ("co_occurrence", "cooccurrence", "cooccur", "co_occur"):
        if not has_user_sequences:
            return dict(raw_freqs), dict(raw_freqs)
        scores = {item: float(len(cooccur_sets.get(item, set()))) for item in raw_freqs}
        return scores, dict(raw_freqs)

    if canonical_signal in ("cf_density", "cf_isolation"):
        if not cf_emb_file:
            raise ValueError("signal='cf_density' requires cf_emb_file to be specified.")
        scores = compute_cf_density_scores(cf_emb_file, top_k=top_k_cf)
        return scores, dict(raw_freqs)

    valid_signals = {"frequency", "user_entropy", "pagerank", "target", "co_occurrence", "cf_density"}
    raise ValueError(
        f"Unknown collaborative signal '{signal}'. Valid signals are: {sorted(valid_signals)}"
    )


def truncate_indices(
    indices,
    min_length=1,
    max_length=4,
    allow_base_collisions=True,
    strategy="shortest_unique",
    item_frequencies=None,
    item_scores=None,
    residuals=None,
    residual_threshold=0.2,
):
    """Truncate each item according to the selected variable-length strategy.

    Strategies:
      - 'shortest_unique': Truncate to the shortest catalog-unique prefix in [min_length, max_length].
      - 'popularity' / 'collaborative': Partition items into length tiers based on empirical
                        interaction scores (frequency, user entropy, PageRank, etc.).
                        High-scoring items can truncate down to min_length; rare tail items
                        are preserved at max_length.
      - 'residual': Truncate to the shortest unique prefix whose cumulative reconstruction error
                    is at or below residual_threshold.
    """
    valid_strategies = {"shortest_unique", "popularity", "collaborative", "residual"}
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
    if strategy in ("popularity", "collaborative"):
        ranking_scores = item_scores if item_scores is not None else item_frequencies
        if ranking_scores is None:
            raise ValueError(
                f"strategy='{strategy}' requires item_frequencies or item_scores."
            )
        k_levels = max_length - min_length + 1
        if k_levels <= 1:
            for item_id in indices:
                item_min_len[item_id] = max_length
        else:
            sorted_items = sorted(
                indices.keys(), key=lambda k: (float(ranking_scores.get(str(k), 0.0)), str(k))
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

        elif strategy in ("popularity", "collaborative"):
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
        choices=["shortest_unique", "popularity", "collaborative", "residual"],
        default="shortest_unique",
        help="Variable-length truncation strategy (default: shortest_unique).",
    )
    parser.add_argument(
        "--collab-signal",
        "--popularity-signal",
        choices=["frequency", "user_entropy", "pagerank", "target", "co_occurrence", "cf_density"],
        default="frequency",
        dest="collab_signal",
        help="Collaborative signal for popularity/collaborative strategy: frequency, user_entropy, pagerank, target, co_occurrence, cf_density (default: frequency).",
    )
    parser.add_argument(
        "--inter-file",
        type=str,
        default=None,
        help="Path to interaction JSON file (<dataset>.inter.json or {item: count}) for popularity strategy.",
    )
    parser.add_argument(
        "--cf-emb-file",
        type=str,
        default=None,
        help="Path to precomputed CF embeddings (.pt, .npy, .json) for cf_density signal.",
    )
    parser.add_argument(
        "--cf-k-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors to evaluate for CF crowd density (default: 10).",
    )
    parser.add_argument(
        "--pagerank-damping",
        type=float,
        default=0.85,
        help="Damping factor for PageRank transition centrality (default: 0.85).",
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

    if args.strategy in ("popularity", "collaborative"):
        if args.collab_signal == "cf_density" and not args.cf_emb_file:
            parser.error("--collab-signal cf_density requires --cf-emb-file.")
        elif args.collab_signal != "cf_density" and not args.inter_file:
            parser.error(
                f"--strategy {args.strategy} requires --inter-file (or --cf-emb-file for cf_density)."
            )
    if args.strategy == "residual" and not args.residuals_file:
        parser.error("--strategy residual requires --residuals-file.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open(encoding="utf-8") as input_file:
        indices = json.load(input_file)

    item_scores = None
    item_frequencies = None
    if args.inter_file or (
        args.strategy in ("popularity", "collaborative") and args.collab_signal == "cf_density"
    ):
        item_scores, item_frequencies = compute_interaction_signals(
            inter_source=args.inter_file if args.inter_file else {},
            signal=args.collab_signal,
            cf_emb_file=args.cf_emb_file,
            top_k_cf=args.cf_k_neighbors,
            damping=args.pagerank_damping,
        )

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
        item_scores=item_scores,
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

    if args.strategy in ("popularity", "collaborative"):
        summary_payload["inter_file"] = str(args.inter_file) if args.inter_file else None
        summary_payload["collab_signal"] = args.collab_signal
        if args.cf_emb_file:
            summary_payload["cf_emb_file"] = str(args.cf_emb_file)
        if item_scores:
            score_sum = sum(abs(item_scores.get(str(i), 0.0)) for i in truncated)
            if score_sum > 0:
                summary_payload["signal_weighted_mean_length"] = sum(
                    lengths[i] * item_scores.get(str(i), 0.0) for i in truncated
                ) / score_sum
        if item_frequencies:
            freq_sum = sum(item_frequencies.get(str(i), 0) for i in truncated)
            if freq_sum > 0:
                summary_payload["traffic_weighted_mean_length"] = sum(
                    lengths[i] * item_frequencies.get(str(i), 0) for i in truncated
                ) / freq_sum
                summary_payload["weighted_mean_length"] = summary_payload["traffic_weighted_mean_length"]
    elif args.strategy == "residual":
        summary_payload["residuals_file"] = str(args.residuals_file)
        summary_payload["residual_threshold"] = args.residual_threshold

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            summary_payload,
            summary_file,
            indent=2,
        )

    print(
        f"Saved {len(truncated)} variable-length IDs ({args.strategy}:{args.collab_signal if args.strategy in ('popularity', 'collaborative') else 'none'}) to {output_path}"
    )
    print(f"Saved length summary to {summary_path}")


if __name__ == "__main__":
    main()


import datetime
import os


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def disambiguate_indices(indices_dict, token_prefix="<dis_{}>"):
    """Appends terminal disambiguation tokens to guarantee zero collisions.

    - Unique items receive index 0 (<dis_0>).
    - Colliding items receive sequential indices 1..N sorted by raw item index ascending.

    Args:
        indices_dict: dict mapping item_id (str or int) to list/tuple of tokens.
        token_prefix: format string for disambiguation token, default "<dis_{}>".

    Returns:
        dict: mapping item_id to list of tokens with disambiguation token appended.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for item_id, tokens in indices_dict.items():
        if tokens and isinstance(tokens[-1], str) and tokens[-1].startswith("<dis_"):
            clean_tokens = tokens[:-1]
        else:
            clean_tokens = tokens
        groups[tuple(clean_tokens)].append(str(item_id))

    resolved = {}
    for code_tuple, items in groups.items():
        if len(items) == 1:
            resolved[items[0]] = list(code_tuple) + [token_prefix.format(0)]
        else:
            items.sort(key=lambda it: int(it) if str(it).isdigit() else str(it))
            for seq_idx, item_id in enumerate(items, start=1):
                resolved[item_id] = list(code_tuple) + [token_prefix.format(seq_idx)]

    ordered_resolved = {}
    for item_id in indices_dict:
        k = str(item_id)
        if k in resolved:
            ordered_resolved[item_id] = resolved[k]
        elif item_id in resolved:
            ordered_resolved[item_id] = resolved[item_id]
        else:
            ordered_resolved[item_id] = resolved.get(k)
    return ordered_resolved





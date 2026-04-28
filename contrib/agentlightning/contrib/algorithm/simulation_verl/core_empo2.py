import torch
from typing import List, Any

def is_sublist(sub, full):
    n, m = len(sub), len(full)
    return any(full[i:i+n] == sub for i in range(m - n + 1))

# Function to remove segments of a list between a start pattern and an end pattern
def remove_pattern_ranges(
    seq: List[Any],
    start_pat_1: List[Any],  # encode(".\n\n<tip>")
    start_pat_2: List[Any],  # encode("<tip>")
    end_pat: List[Any],       # encode("</tip>\n\n")
    dot_token: List[Any],     # encode(".")
) -> List[Any]:
    """
    Remove every [start ... end] slice from seq.
    The last match uses start_pat_2, all others use start_pat_1.
    For non-last matches, the leading "." is re-inserted after removal
    since ".\n\n" is a single token that can't be split.
    """

    # --- Pass 1: Find all (start, end) ranges ---
    # Try the longer pattern (start_pat_1) first, fall back to start_pat_2
    ranges: List[tuple] = []
    i = 0
    n = len(seq)
    ls1, ls2, le = len(start_pat_1), len(start_pat_2), len(end_pat)

    while i < n:
        matched_start = -1
        matched_ls = 0
        if i + ls1 <= n and seq[i:i+ls1] == start_pat_1:
            matched_start = i
            matched_ls = ls1
        elif i + ls2 <= n and seq[i:i+ls2] == start_pat_2:
            matched_start = i
            matched_ls = ls2

        if matched_start != -1:
            j = matched_start + matched_ls
            while j + le <= n:
                if seq[j:j+le] == end_pat:
                    ranges.append((matched_start, j + le))  # [start, end) exclusive
                    i = j + le
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1

    if not ranges:
        return list(seq)

    # --- Pass 2: Determine final removal ranges and insertions ---
    # Non-last matches: remove ".\n\n<tip>...</tip>\n\n", re-insert "." token
    # Last match: preserve ".\n\n", remove only "<tip>...</tip>\n\n"
    final_ranges: List[tuple] = []
    insertions: dict = {}  # {range_start: tokens to insert at that position}

    for idx, (s, e) in enumerate(ranges):
        if idx < len(ranges) - 1:
            # Non-last match: must begin with start_pat_1 (".\n\n<tip>")
            if seq[s:s+ls1] == start_pat_1:
                final_ranges.append((s, e))
                insertions[s] = dot_token  # Re-insert "." after removal
            # If it doesn't match start_pat_1, skip (not a valid removal target)
        else:
            # Last match: use start_pat_2 ("<tip>") boundary
            # If it matched start_pat_1, preserve the leading ".\n\n" token(s)
            if seq[s:s+ls1] == start_pat_1:
                diff = ls1 - ls2
                final_ranges.append((s + diff, e))
            else:
                final_ranges.append((s, e))

    # --- Pass 3: Rebuild sequence, skipping removal ranges ---
    out: List[Any] = []
    i = 0
    r = 0
    while i < n:
        if r < len(final_ranges) and i == final_ranges[r][0]:
            # Insert replacement tokens (e.g. ".") if any
            if i in insertions:
                out.extend(insertions[i])
            i = final_ranges[r][1]
            r += 1
        else:
            out.append(seq[i])
            i += 1

    return out

def low_prob_token_masking(batch):
    response_mask = batch.batch["response_mask"]       # [N, T]
    if "old_log_prob_off_policy" in batch.batch:
        old_log_prob = batch.batch["old_log_prob_off_policy"]
    else:
        old_log_prob = batch.batch["old_log_probs"]        # [N, T]
    # advantages = batch.batch["advantages"]             # [N, T]

    masked_old_log_prob = old_log_prob.masked_fill(response_mask == 0, 1e9)
    min_values, _ = torch.min(masked_old_log_prob, dim=1)  # [N]

    mask = min_values < -5  # [N]

    combined_mask = mask.unsqueeze(1) & (response_mask == 1)

    # advantages masking
    response_mask = response_mask.masked_fill(combined_mask, 0)
    batch.batch["response_mask"] = response_mask

    print(f"Number of tokens masked: {combined_mask.sum().item()}")

    return batch

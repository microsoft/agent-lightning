import torch
from typing import List, Any

def is_sublist(sub, full):
    n, m = len(sub), len(full)
    return any(full[i:i+n] == sub for i in range(m - n + 1))

# Function to remove segments of a list between a start pattern and an end pattern
def remove_pattern_ranges(seq: List[Any],
                        start_pat: List[Any],
                        end_pat: List[Any]) -> List[Any]:
    """Remove every [start_pat ... end_pat] slice (inclusive) from seq."""
    
    out: List[Any] = []
    i = 0
    n = len(seq)
    ls, le = len(start_pat), len(end_pat)

    while i < n:
        # Check if the start pattern matches at the current position
        if i + ls <= n and seq[i:i+ls] == start_pat:
            # Look for the first occurrence of the end pattern after the start pattern
            j = i + ls
            found_end = -1
            while j + le <= n:
                if seq[j:j+le] == end_pat:
                    found_end = j
                    break  # Stop when the end pattern is found
                j += 1

            # If the end pattern is found, skip the whole segment from start to end
            if found_end != -1:
                i = found_end + le  # Move the index past the end pattern
                continue  # Skip the current iteration and go to the next
            else:
                # If the end pattern is not found, keep the current element and move one step forward
                out.append(seq[i])
                i += 1
        else:
            # If the start pattern is not found, just append the current element
            out.append(seq[i])
            i += 1

    # Return the filtered list with the start-end pattern segments removed
    return out

def low_prob_token_masking(batch):
    response_mask = batch.batch["response_mask"]       # [N, T]
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
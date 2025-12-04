import tokenizers
from transformers import AutoTokenizer
model_dir = 'meta-llama/Llama-3.2-3B-Instruct'
tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)

count = 0

def find_min_diff_lengths(list_a, list_b):
    """
    计算两个列表中不匹配的连续子序列的最少长度。
    这通过找出最长公共子序列 (LCS) 实现，然后返回两个列表中非 LCS 部分的长度。

    Args:
        list_a (list): 第一个列表。
        list_b (list): 第二个列表。

    Returns:
        tuple: 一个元组 (diff_a_len, diff_b_len)，
               分别表示 list_a 和 list_b 中与 LCS 不匹配部分的长度。
    """
    len_a = len(list_a)
    len_b = len(list_b)

    # 1. 初始化动态规划表 (DP Table)
    # dp[i][j] 存储 list_a[:i] 和 list_b[:j] 的 LCS 长度
    # 尺寸为 (len_a + 1) x (len_b + 1)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    # 2. 填充 DP 表
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if list_a[i - 1] == list_b[j - 1]:
                # 如果当前元素匹配，LCS 长度加 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 如果不匹配，取 (排除 list_a[i-1]) 和 (排除 list_b[j-1]) 中的较大 LCS 长度
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 3. 获取最长公共子序列 (LCS) 的长度
    lcs_length = dp[len_a][len_b]

    # 4. 计算差异部分的长度
    # 差异部分长度 = 总长度 - LCS 长度
    diff_a_len = len_a - lcs_length
    diff_b_len = len_b - lcs_length

    return (diff_a_len, diff_b_len)

# # --- 示例调用 ---
# list_1 = [1, 2, 3, 5, 6, 9]
# list_2 = [1, 2, 4, 3, 9]

# result = find_min_diff_lengths(list_1, list_2)
# print(f"列表 A: {list_1}")
# print(f"列表 B: {list_2}")
# print(f"差异部分的最少长度 (A, B): {result}")

# # 示例 2
# list_a = ["A", "B", "C", "D", "E"]
# list_b = ["A", "F", "D", "E"]
# # LCS: ["A", "D", "E"] -> 长度 3
# # A 中非 LCS: ["B", "C"] -> 长度 2
# # B 中非 LCS: ["F"] -> 长度 1
# result_2 = find_min_diff_lengths(list_a, list_b)
# print("\n--- 示例 2 ---")
# print(f"列表 A: {list_a}")
# print(f"列表 B: {list_b}")
# print(f"差异部分的最少长度 (A, B): {result_2}")

def logged_startswith(full_ids, prefix_ids, tokenizer):
    template_mismatch, retoken_mismatch, others_mismatch = False, False, False
    if full_ids[:len(prefix_ids)] == prefix_ids:
        merge = True
        return template_mismatch, retoken_mismatch, others_mismatch, merge
    else:
        merge = False

    def _special_token_sequence(ids):
        return [id for id in ids if id in tokenizer.all_special_ids]
    
    def _none_special_token_sequence(ids):
        return [id for id in ids if id not in tokenizer.all_special_ids]

    # First, handle special tokens
    full_special_ids = _special_token_sequence(full_ids)
    prefix_special_ids = _special_token_sequence(prefix_ids)
    if len(full_special_ids) != len(prefix_special_ids) or sum(1 for a, b in zip(full_special_ids, prefix_special_ids) if a != b) > 0:
        template_mismatch = True

    # Next, handle string content
    full_content_ids = _none_special_token_sequence(full_ids)
    prefix_content_ids = _none_special_token_sequence(prefix_ids)
    full_string = tokenizer.decode(full_ids, skip_special_tokens=True)
    prefix_string = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    if full_content_ids[:len(prefix_content_ids)] != prefix_content_ids and full_string.startswith(prefix_string):
        retoken_mismatch = True
        # diff_segments = find_all_diff_segments(full_content_ids[:len(prefix_content_ids)], prefix_content_ids)
        # if len(diff_segments) == 1:
        #     diff_a, diff_b = diff_segments[0]
        #     diff_string_a = tokenizer.decode(diff_a, skip_special_tokens=True)
        #     diff_string_b = tokenizer.decode(diff_b, skip_special_tokens=True)
        #     if diff_string_a == "<think" and diff_string_b == "<think":
        #         pass
        #     else:
        #         with open("count.log", "a+") as f:
        #             print(1, file=f)
        # else:
        #     with open("count.log", "a+") as f:
        #             print(1, file=f)


        # for diff_a, diff_b in diff_segments:
        #     diff_string_a = tokenizer.decode(diff_a, skip_special_tokens=True)
        #     diff_string_b = tokenizer.decode(diff_b, skip_special_tokens=True)
        #     import pdb; pdb.set_trace()
        #     if diff_string_a == "<think":
        #         with open("count.log", "a+") as f:
        #             print(1, file=f)
        #     # if not diff_string_a == "<think" and not diff_string_b == "<think":
        #     #     # import pdb; pdb.set_trace()
        #     #     with open("retoken_mismatch.log", "a+") as f:
        #     #         print(f"{diff_a}, {diff_b}, \"{diff_string_a}\", \"{diff_string_b}\"", file=f)

        a, b = find_min_diff_lengths(full_content_ids[:len(prefix_content_ids)], prefix_content_ids)
        with open("count.log", "a+") as f:
            print(a, b, len(full_content_ids), len(prefix_content_ids), file=f)
    else:
        import pdb; pdb.set_trace()
    # elif full_content_ids[:len(prefix_content_ids)] != prefix_content_ids and not full_string.startswith(prefix_string):
    #     others_mismatch = True
    # elif full_content_ids[:len(prefix_content_ids)] == prefix_content_ids:
    #     # case 1: fully match; case 2: special token mismatch only
    #     # case 1: template_mismatch == False, retoken_mismatch == False, others_mismatch == False, merge == True
    #     # case 2: template_mismatch == True, retoken_mismatch == False, others_mismatch == False, merge == False
    #     if not ((not template_mismatch and not retoken_mismatch and not others_mismatch and merge) \
    #         or (template_mismatch and not retoken_mismatch and not others_mismatch and not merge)):
    #         with open("bad_case_jiahang.log", "a+") as f:
    #             print("-" * 20, file=f)
    #             print("full_ids:", file=f)
    #             print(full_ids, file=f)
    #             print("prefix_ids:", file=f)
    #             print(prefix_ids, file=f)
    #             print(f"template_mismatch: {template_mismatch}, retoken_mismatch: {retoken_mismatch}, others_mismatch: {others_mismatch}, merge: {merge}", file=f)
    return template_mismatch, retoken_mismatch, others_mismatch, merge




def fuzzy_startswith(full_ids, prefix_ids, tokenizer, special_token_tolerance=0, string_tolerance=0):
    def _special_token_sequence(ids):
        return [id for id in ids if id in tokenizer.all_special_ids]

    if special_token_tolerance < 0 or string_tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    # First, handle special tokens
    full_special_ids = _special_token_sequence(full_ids)
    prefix_special_ids = _special_token_sequence(prefix_ids)
    diff_count = sum(1 for a, b in zip(full_special_ids, prefix_special_ids) if a != b)
    special_token_tolerance -= diff_count
    if special_token_tolerance < 0:
        return diff_count, False

    # Next, handle string content
    full_string = tokenizer.decode(full_ids, skip_special_tokens=True)
    prefix_string = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    full_ids = tokenizer.encode(full_string)
    prefix_ids = tokenizer.encode(prefix_string)
    full_string = tokenizer.decode(full_ids, skip_special_tokens=True)
    prefix_string = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    m = len(prefix_string)
    n = len(full_string)

    if m == 0:
        return True  # Empty B always matches (distance 0 to empty prefix)
    if n == 0:
        return m <= string_tolerance  # B non-empty but A empty: only match if we can delete all of B within tolerance
    if string_tolerance == 0:
        return full_string.startswith(prefix_string)  # exact match required

    # use DP to compute edit distance with banded optimization
    min_j = max(0, m - string_tolerance)
    max_j = min(n, m + string_tolerance)
    if min_j > max_j:
        return False  # no possible prefix length

    prev_start = max(0, 0 - string_tolerance)
    prev_end = min(n, 0 + string_tolerance)
    prev = [j for j in range(prev_start, prev_end + 1)]

    for j_idx, j in enumerate(range(prev_start, prev_end + 1)):
        if min_j <= j <= max_j and prev[j_idx] <= string_tolerance:
            return True

    for i in range(1, m + 1):
        # valid j range for this row
        start_j = max(0, i - string_tolerance)
        end_j = min(n, i + string_tolerance)
        cur_len = end_j - start_j + 1
        cur = [0] * cur_len

        for idx, j in enumerate(range(start_j, end_j + 1)):
            del_cost = None
            prev_start = max(0, (i - 1) - string_tolerance)
            prev_end = min(n, (i - 1) + string_tolerance)
            if prev_start <= j <= prev_end:
                del_cost = prev[j - prev_start] + 1
            else:
                del_cost = abs((i - 1) - j) + 1  # safe over-approximation

            ins_cost = None
            if j - 1 >= start_j:
                ins_cost = cur[idx - 1] + 1
            else:
                ins_cost = abs(i - (j - 1)) + 1

            sub_cost = None
            if prev_start <= (j - 1) <= prev_end:
                sub_cost = prev[(j - 1) - prev_start] + (0 if prefix_string[i - 1] == full_string[j - 1] else 1)
            else:
                sub_cost = abs((i - 1) - (j - 1)) + (0 if prefix_string[i - 1] == full_string[j - 1] else 1)

            cur[idx] = min(del_cost, ins_cost, sub_cost)

        for idx, j in enumerate(range(start_j, end_j + 1)):
            if min_j <= j <= max_j and cur[idx] <= string_tolerance:
                return True
        prev = cur
    return False


def find_all_diff_segments(a, b):
    i = j = 0
    n1, n2 = len(a), len(b)
    diffs = []
    curr_a, curr_b = [], []

    while i < n1 or j < n2:
        # 如果两个列表都没结束并且元素相同 → diff 结束（如果在记录）
        if i < n1 and j < n2 and a[i] == b[j]:
            if curr_a or curr_b:
                diffs.append((curr_a, curr_b))
                curr_a, curr_b = [], []
            i += 1
            j += 1
            continue

        # 下面是元素不同的情况，需要归类到 diff
        if i < n1:
            curr_a.append(a[i])
        if j < n2:
            curr_b.append(b[j])
        i += 1 if i < n1 else 0
        j += 1 if j < n2 else 0

    # 结束时如果还有 diff 段
    if curr_a or curr_b:
        diffs.append((curr_a, curr_b))

    return diffs


import re
from typing import List, Dict, Any, Tuple

def parse_step_data(lines: List[str]) -> List[Tuple[List[float], List[float]]]:
    res = []
    def str_to_float_list(s: str) -> List[float]:
        numbers = re.findall(r'(\d+\.?\d*)', s)
        return [int(n) for n in numbers]
    data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('---')]
    for i in range(0, len(data_lines), 2):
        res.append((str_to_float_list(data_lines[i]), str_to_float_list(data_lines[i+1])))
    return res

def process_txt_file(file_path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    current_step = -1
    current_step_lines: List[str] = []
    
    for line in content:
        line = line.strip()
        step_match = re.match(r'---------- Step (\d+)----------', line)
        if step_match:
            if current_step != -1:
                step_key = f"step_{current_step}"
                data[step_key] = parse_step_data(current_step_lines)
            
            current_step = int(step_match.group(1))
            current_step_lines = []
            continue
            
        if current_step == -1 and line:
            if line.startswith('[') or line.startswith('---'):
                current_step = 0
        
        if current_step != -1:
            current_step_lines.append(line)

    if current_step != -1:
        step_key = f"step_{current_step}"
        data[step_key] = parse_step_data(current_step_lines)
        
    return data

result_data = process_txt_file("/home/jiahangxu/teamdrive/search_r1/mismatch_logs/Llama-3.2-3B-Instruct/searchr1_minibatch256_runner32_trajectory_synced/retoken_mismatch.log")

for step, values in result_data.items():
    print("----------", step, "--------", len(values))
    with open("retoken_mismatch.log", "a+") as f:
        print("----------", step, "--------", len(values), file=f)
    for item in values:
        record = logged_startswith(item[0], item[1], tok)
        # print(record[1], record[3])
        # assert record[1] == True and record[3] == False
        if not (record[1] == True and record[3] == False):
            import pdb; pdb.set_trace()
            print("Mismatch found:")
            print("Full IDs: ", item[0])
            print("Prefix IDs: ", item[1])
            print(f"template_mismatch: {record[0]}, retoken_mismatch: {record[1]}, others_mismatch: {record[2]}, merge: {record[3]}")
    print("Finished step:", step)
    import pdb; pdb.set_trace()

    # TODO: for retoken mismatch,计算所有不同的diff tokens的数量占整体数量的比例
    
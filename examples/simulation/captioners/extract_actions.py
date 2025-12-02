import re
def extract_pure_action(llm_output):
    match = re.search(r"<[Aa]ction>(.*?)</[Aa]ction>", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return llm_output[-30:], False

def extract_reasoning(llm_output):
    match = re.search(r"<[Tt]hink>(.*?)</[Tt]hink>", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return None, False
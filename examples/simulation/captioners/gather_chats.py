def gather_chats(obs):
    chat_list = []
    for item in obs:
        role = item.type
        content = item.content
        if "System" in role:
            continue
        elif "User" in role:
            role = "user"
        else:
            role = "assistant"
        chat_list.append(f"{role}: {content}")
    text = " ".join(chat_list)
    return text

def diff_strings(new_token, old_token):
    operations = []
    i, j = 0, 0  # 初始化指针

    while i < len(new_token) and j < len(old_token):
        if new_token[i] == old_token[j]:
            # 当前字符相同，指针同时向前移动
            i += 1
            j += 1
        else:
            # 当前字符不同，判断操作类型
            if i < len(new_token) and new_token[i] not in old_token[j:]:
                # A中的字符不在B的剩余部分中，需要删除
                operations.append((i, "delete", new_token[i]))
                i += 1
            elif j < len(old_token) and old_token[j] not in new_token[i:]:
                # B中的字符不在A的剩余部分中，需要插入
                operations.append((j, "insert", old_token[j]))
                j += 1
            else:
                # 当前字符不同，但都在对方剩余部分中，需要替换
                operations.append((i, "replace", new_token[i], old_token[j]))
                i += 1
                j += 1

    # 处理剩余的字符
    while i < len(new_token):
        operations.append((i, "delete", new_token[i]))
        i += 1

    while j < len(old_token):
        operations.append((j, "insert", old_token[j]))
        j += 1

    return operations


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    # 示例
    
    old_str = "My name is John Doe"
    new_str = "My name is Jane Smith"
    token_old_str = tokenizer.encode(old_str)
    token_new_str = tokenizer.encode(new_str)
    operations = diff_strings(token_new_str,token_old_str)
    print("Operations:")
    for op in operations:
        print(f"Position {op[0]}: {op[1]}")
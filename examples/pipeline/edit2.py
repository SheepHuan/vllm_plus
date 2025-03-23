def rabin_karp_search_optimized(token_list1, token_list2, window_size):
    def compute_initial_hash(tokens, length):
        # 计算第一个窗口的哈希值
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value * base + tokens[i]) % modulus
        return hash_value

    def roll_hash(prev_hash, prev_token, next_token, base_power):
        # 滚动计算下一个窗口的哈希值
        hash_value = (prev_hash - prev_token * base_power) % modulus
        hash_value = (hash_value * base + next_token) % modulus
        return hash_value

    base = 256  # 基数，通常选择一个质数
    modulus = 101  # 模数，选择一个质数

    n1 = len(token_list1)
    n2 = len(token_list2)
    matches = []

    # 预计算 base^(window_size - 1)
    base_power = pow(base, window_size - 1, modulus)

    # 计算token_list1中所有窗口的哈希值
    hash_list1 = []
    if n1 >= window_size:
        hash_value = compute_initial_hash(token_list1, window_size)
        hash_list1.append(hash_value)
        for i in range(1, n1 - window_size + 1):
            hash_value = roll_hash(hash_value, token_list1[i - 1], token_list1[i + window_size - 1], base_power)
            hash_list1.append(hash_value)

    # 计算token_list2中所有窗口的哈希值
    hash_list2 = []
    if n2 >= window_size:
        hash_value = compute_initial_hash(token_list2, window_size)
        hash_list2.append(hash_value)
        for i in range(1, n2 - window_size + 1):
            hash_value = roll_hash(hash_value, token_list2[i - 1], token_list2[i + window_size - 1], base_power)
            hash_list2.append(hash_value)

    # 比较哈希值
    for i in range(len(hash_list1)):
        for j in range(len(hash_list2)):
            if hash_list1[i] == hash_list2[j]:
                # 哈希值相同，进一步比较内容
                if token_list1[i:i + window_size] == token_list2[j:j + window_size]:
                    matches.append(((i, i + window_size - 1), (j, j + window_size - 1)))

    return matches

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# 示例
token_list1 = tokenizer.encode("I like eating apple, but my brother likes eating banana.")
token_list2 = tokenizer.encode("I like eating apple, but my sister likes eating banana.")
window_size = 3

matches = rabin_karp_search_optimized(token_list1, token_list2, window_size)
print(matches)
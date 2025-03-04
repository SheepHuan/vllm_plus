from transformers import AutoTokenizer
import copy


def edit_distance_with_operations(str1, str2,tokenizer):
    """
    1. 可优化的：
    1. 相同词义的token可以不替换，例如ok ,OK
    2. 
    
    """
    m, n = len(str1), len(str2)
    # 创建一个二维数组 dp 用于存储子问题的解
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化第一行和第一列
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充 dp 数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )

    # 回溯找出所有编辑操作
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i - 1] == str2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(["Delete",str1[i - 1],i - 1])
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(["Insert",str2[j - 1],i])
            j -= 1
        elif i > 0 and j > 0:
            operations.append(["Replace",str1[i - 1],str2[j - 1],i - 1])
            i -= 1
            j -= 1

    # 反转操作列表，使其按操作顺序排列
    operations.reverse()

    return dp[m][n], operations


def transform_operations(str1,operations):
    result = copy.deepcopy(str1)    
    for op in operations:
        if op[0]=="Delete":
            index = int(op[-1])
            del result[index]
        elif op[0]=="Insert":
            char = op[1]
            index = int(op[-1])
            result.insert(index, char)
        elif op[0]=="Replace":
            new_char = op[2]
            index = int(op[-1])
            result[index] = new_char
    return result



if __name__ == "__main__":
    text1 = "Ok, can you again summarize the different interventions that could addresses each of the key components of the COM-B model and gets people to fill in the questionnaire?"
    text2 = "Ok, how would a comprehensive intervention look like that addresses each of the key components of the COM-B model and gets people to fill in the questionnaire?"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)

    
    distance, operations = edit_distance_with_operations(tokens1, tokens2,tokenizer)
    result = transform_operations(tokens1,operations)
    print(distance)
    print(operations)
    print(tokenizer.decode(result))
    print(tokenizer.decode(tokens2))

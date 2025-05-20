import json
import tiktoken

def convert_content_to_string(content):
    """将 content 转换为字符串格式"""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        # 如果是字典，将所有值拼接成一个字符串
        return " ".join(str(value) for value in content.values())
    elif isinstance(content, list):
        # 如果是列表，将所有元素拼接成一个字符串
        return " ".join(str(item) for item in content)
    else:
        # 其他类型（如数字、布尔值等），直接转换为字符串
        return str(content)

def calculate_token_length(jsonl_file, encoding):
    """计算 JSONL 文件中每个 content 字段的 token 长度"""
    token_lengths = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            
            # 将 content 转换为字符串
            content_str = convert_content_to_string(content)
            
            # 使用 encoding 计算 token 长度
            tokens = encoding.encode(content_str)
            token_lengths.append(len(tokens))
    
    return token_lengths

# 加载 tiktoken 编码器
# encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.get_encoding("o200k_base")
# 或者使用以下代码来获取 GPT-4 的编码器
# encoding = tiktoken.encoding_for_model("gpt-4")

# JSONL 文件路径
jsonl_file = "E:\\wanlong\\codes\\rag_table\\datasets\\med_top500_new.jsonl"

# 计算 token 长度
token_lengths = calculate_token_length(jsonl_file, encoding)

# 输出结果
for idx, length in enumerate(token_lengths):
    print(f"Line {idx + 1}: {length} tokens")

# 计算 token 长度的总和
total_tokens = sum(token_lengths)
print(f"Total tokens: {total_tokens}")
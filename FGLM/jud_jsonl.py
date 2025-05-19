import json

def validate_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = 0
            for line in file:
                line_number += 1
                # 去除空白字符
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 尝试解析JSON对象
                try:
                    json_object = json.loads(line)
                    print(f"Line {line_number}: Valid JSON object - ")
                except json.JSONDecodeError as e:
                    print(f"Error on line {line_number}: Invalid JSON - {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
file_path = "./datasets/fin_top302.jsonl"  # 替换为你的jsonl文件路径
validate_jsonl(file_path)
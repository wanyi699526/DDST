import json
import re

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                content = data.get('content', {})
                
                # 处理content字典中的每一篇文章
                for key in content:
                    article = content[key]
                    # Find the first occurrence of "用法用量"
                    start_idx = article.find('【用法用量】')
                    if start_idx == -1:
                        continue
                    
                    # Find the last occurrence of "药代动力学"
                    end_idx = article.rfind('【药代动力学】')
                    if end_idx == -1:
                        continue
                    
                    # Remove content between the markers
                    new_content = article[:start_idx] + article[end_idx:]
                    content[key] = new_content
                
                # 更新content字段
                data['content'] = content
                
                # Write the modified data to output file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line}")
                continue

if __name__ == "__main__":
    input_file = "E:\\wanlong\\codes\\rag_table\\datasets\\med_top500.jsonl"  # 输入文件名
    output_file = "E:\\wanlong\\codes\\rag_table\\datasets\\med_top500_new.jsonl"  # 输出文件名
    process_jsonl(input_file, output_file)
from FlagEmbedding import FlagReranker
import json

# 初始化reranker
reranker = FlagReranker('./models/embedding/bge-reranker-v2-m3', use_fp16=True)
# score = reranker.compute_score(['金融','法律'],normalize=True)

# 从jsonl文件中读取数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 计算文章中任意两篇文章之间的相关性得分
def calculate_article_similarities(data):
    # 获取所有文章的内容
    articles = list(data["content"].values())
    
    # 存储相关性得分的字典
    similarity_scores = {}
    
    # 遍历所有文章对
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            # 计算相关性得分
            score = reranker.compute_score([articles[i], articles[j]], normalize=True)
            # 将得分存储在字典中，键为文章对的编号
            similarity_scores[f"{i+1}-{j+1}"] = score
    
    return similarity_scores

# 主函数
def main():
    # 加载jsonl文件
    file_path = './test/lda_test.jsonl'  # 替换为你的jsonl文件路径
    data_list = load_jsonl(file_path)
    
    # 遍历每个数据对象
    for data in data_list:
        print(f"Processing data with ID: {data['id']}")
        
        # 计算相关性得分
        similarity_scores = calculate_article_similarities(data)
        
        # 打印相关性得分
        for pair, score in similarity_scores.items():
            print(f"文章 {pair} 之间的相关性得分为: {score}")
        print("\n")  # 分隔不同数据对象的输出

if __name__ == "__main__":
    # pass
    main()
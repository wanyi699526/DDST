import json
import numpy as np
import bert_score
import argparse
import os
from tqdm import tqdm
import copy
import torch
import functools
import datetime
from sacrebleu import sentence_chrf
import traceback
import logging
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bleurt import score as bleurt_score
from bart_score import BARTScorer

# 配置参数 - 直接在代码中指定
INPUT_FILE = "data/test_output.jsonl"  # 替换为您的实际输入文件路径
MODEL_NAME = "gpt-4"  # 替换为您的实际模型名称
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 50  # 每处理多少个样本保存一次检查点
MAX_RETRIES = 3  # 每个样本的最大重试次数

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# 使用缓存装饰器来加速重复计算
@functools.lru_cache(maxsize=10000)
def cached_bert_score(pred_text, target_text):
    """缓存BERT得分计算，避免重复计算相同的文本对"""
    global bert_scorer
    try:
        _, _, f1 = bert_scorer.score([pred_text], [target_text])
        return max(f1.item(), 0)
    except:
        return 0.0

# 全局变量，只初始化一次
bert_scorer = None

def init_bert_scorer():
    """初始化BERTScorer，避免重复加载模型"""
    global bert_scorer
    if bert_scorer is None:
        logging.info("初始化 BERT Scorer 模型...")
        bert_scorer = bert_score.BERTScorer(model_type="bert-base-chinese", lang="zh", rescale_with_baseline=True)
        # 确保模型在GPU上运行（如果可用）
        if torch.cuda.is_available():
            logging.info("使用GPU加速计算")
        else:
            logging.info("未检测到GPU，使用CPU计算（较慢）")
    return bert_scorer

def column_to_row_format(table):
    """将列式存储的表格转换为行式存储"""
    rows = []
    columns = list(table.keys())
    num_rows = len(table[columns[0]]) if columns else 0
    
    for i in range(num_rows):
        row = {}
        for col in columns:
            # 确保使用字符串类型的键
            row[col] = str(table[col][i]) if i < len(table[col]) and table[col][i] is not None else ""
        rows.append(row)
    return rows

def flatten_table(table):
    """将表格展平为一个集合，每个元素是 (row_idx, col, value) 的元组，None 值也包含在内"""
    flattened = set()
    for row_idx, row_data in enumerate(table):
        for col, value in row_data.items():
            # 统一处理空值
            if value is None or value == "None":
                value = "None"
            # 处理列表类型的值，将其转换为字符串
            elif isinstance(value, list):
                value = str(value)
            # 确保使用字符串类型
            else:
                value = str(value)
            flattened.add((row_idx, col, value))
    return flattened

def calc_similarity_matrix(tgt_data, pred_data, metric):
    """计算相似度矩阵，支持多种度量标准"""
    def calc_data_similarity(tgt, pred):
        # 提取元组中的 value 部分
        tgt_value = tgt[2]  # (row_idx, col, value) 中的 value
        pred_value = pred[2]  # (row_idx, col, value) 中的 value

        # 如果 tgt_value 或 pred_value 为 None，返回默认值 0
        if tgt_value is None or pred_value is None or tgt_value == "None" or pred_value == "None":
            return 0.0

        if metric == 'E':
            return int(tgt_value == pred_value)
        elif metric == 'c':
            # 如果输入为空字符串，返回默认值 0
            if not tgt_value or not pred_value:
                return 0.0
            try:
                return sentence_chrf(pred_value, [tgt_value]).score / 100
            except:
                return 0.0
        elif metric == 'BS-scaled':
            try:
                ret = bert_scorer.score([pred_value], [tgt_value])[2].item()
                return max(ret, 0)
            except:
                return 0.0
        else:
            raise ValueError(f"Metric cannot be {metric}")

    ret_table = []
    for tgt in tgt_data:
        temp = []
        for pred in pred_data:
            temp.append(calc_data_similarity(tgt, pred))
        ret_table.append(temp)
    return np.array(ret_table, dtype=np.float64)

def metrics_by_sim(tgt_data, pred_data, metric):
    """基于相似度矩阵计算 Precision、Recall 和 F1 分数"""
    sim = calc_similarity_matrix(tgt_data, pred_data, metric)
    try:
        prec = np.mean(np.max(sim, axis=0)) if sim.size > 0 else 0
        recall = np.mean(np.max(sim, axis=1)) if sim.size > 0 else 0
    except ValueError:
        prec = 0
        recall = 0

    if prec + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def get_second_column_data(table):
    """获取表格的第二列数据"""
    if not table or len(table) == 0:
        return set()
    
    flattened = set()
    columns = list(table[0].keys())
    if len(columns) >= 2:  # 确保有第二列
        second_col = columns[1]
        for row_idx, row_data in enumerate(table):
            if second_col in row_data:
                value = row_data[second_col]
                if value is None or value == "None":
                    value = "None"  # 保持一致性
                flattened.add((row_idx, second_col, value))
    
    return flattened, second_col

def get_table_headers(table):
    """获取表格的表头"""
    if not table or len(table) == 0:
        return set()
    
    headers = set()
    for col in table[0].keys():
        headers.add(('header', col, col))
    
    return headers

def get_data_cells(table):
    """获取表格的数据单元格（排除第一列和表头）"""
    if not table or len(table) == 0:
        return set()
    
    flattened = set()
    columns = list(table[0].keys())
    if len(columns) > 1:  # 确保有多于一列
        for row_idx, row_data in enumerate(table):
            for col_idx, col in enumerate(columns):
                if col_idx > 0:  # 跳过第一列
                    value = row_data[col]
                    if value is None or value == "None":
                        value = "None"  # 保持一致性
                    flattened.add((row_idx, col, value))
    
    return flattened

def calculate_second_column_f1(true_table, pred_table):
    """计算第二列的F1分数"""
    # 获取真实表的第二列数据
    true_second_col_data, true_second_col_name = get_second_column_data(true_table)
    
    if not true_second_col_data or not true_second_col_name:
        return {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']}
    
    # 获取预测表的第二列数据
    pred_second_col_data, _ = get_second_column_data(pred_table)
    
    if not pred_second_col_data:
        return {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']}
    
    # 计算不同标准下的F1分数
    results = {}
    for metric in ['E', 'c', 'BS-scaled']:
        precision, recall, f1 = metrics_by_sim(true_second_col_data, pred_second_col_data, metric)
        results[metric] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results

def calculate_table_header_f1(true_table, pred_table):
    """计算表头的F1分数"""
    true_headers = get_table_headers(true_table)
    pred_headers = get_table_headers(pred_table)
    
    if not true_headers or not pred_headers:
        return {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']}
    
    # 计算不同标准下的F1分数
    results = {}
    for metric in ['E', 'c', 'BS-scaled']:
        precision, recall, f1 = metrics_by_sim(true_headers, pred_headers, metric)
        results[metric] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results

def calculate_data_cell_f1(true_table, pred_table):
    """计算数据单元格的F1分数"""
    true_data_cells = get_data_cells(true_table)
    pred_data_cells = get_data_cells(pred_table)
    
    if not true_data_cells or not pred_data_cells:
        return {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']}
    
    # 计算不同标准下的F1分数
    results = {}
    for metric in ['E', 'c', 'BS-scaled']:
        precision, recall, f1 = metrics_by_sim(true_data_cells, pred_data_cells, metric)
        results[metric] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results

def calculate_error(true_table, pred_table):
    """计算ERROR指标"""
    # 获取数据单元格
    true_data_cells = get_data_cells(true_table)
    pred_data_cells = get_data_cells(pred_table)
    
    # 1. 收集真实表中的单元格，按(行,列)分组
    true_cells = {}
    for row_idx, col, value in true_data_cells:
        true_cells[(row_idx, col)] = value
    
    # 2. 收集预测表中的单元格，按(行,列)分组
    pred_cells = {}
    for row_idx, col, value in pred_data_cells:
        pred_cells[(row_idx, col)] = value
    
    # 3. 找出预测表中与真实表不一致的单元格数量
    error_count = 0
    for key, true_value in true_cells.items():
        if key in pred_cells:
            pred_value = pred_cells[key]
            if true_value != pred_value and pred_value != "None":
                error_count += 1
    
    # 计算ERROR比率
    error_rate = error_count / len(true_cells) if len(true_cells) > 0 else 0
    return error_rate

def calculate_extra_header_rate(true_table, pred_table):
    """计算额外表头生成率
    
    额外表头生成率 = 额外表头生成字段 / table中表头字段
    额外表头生成字段: 填充数据超过80%且不在table中的表头字段
    """
    # 获取真实表头和预测表头
    true_headers = set(true_table[0].keys()) if true_table else set()
    pred_headers = set(pred_table[0].keys()) if pred_table else set()
    
    # 计算额外生成的表头
    extra_headers = pred_headers - true_headers
    
    # 检查额外表头中的填充率
    extra_headers_with_data = set()
    for header in extra_headers:
        if header in pred_table[0]:
            # 计算该列的非空值比例
            non_null_count = 0
            for row in pred_table:
                if header in row and row[header] not in (None, "None", ""):
                    non_null_count += 1
            
            # 如果填充率超过80%，认为是额外表头生成字段
            if non_null_count / len(pred_table) >= 0.8:
                extra_headers_with_data.add(header)
    
    # 计算额外表头生成率
    extra_header_rate = len(extra_headers_with_data) / len(true_headers) if len(true_headers) > 0 else 0
    
    return extra_header_rate, extra_headers_with_data, true_headers

def evaluate_sample(sample):
    """评估单个样本的所有指标"""
    # 获取真实表格和预测表格
    true_table = sample.get("table", {})
    # 兼容不同的字段名称
    pred_table = sample.get("predict", {})
    if not pred_table:
        pred_table = sample.get("prediction", {})
    
    # 如果两个表格都为空，返回默认值
    if not true_table or not pred_table:
        return {
            "id": sample.get("id", "unknown"),
            "second_column_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
            "table_header_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
            "data_cell_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
            "error_rate": 0,
            "extra_header_rate": 0,
            "extra_header_details": {
                "extra_headers_count": 0,
                "true_headers_count": 0,
                "extra_headers": [],
                "true_headers": []
            }
        }
    
    # 转换为行格式 - 只做一次
    true_table_rows = column_to_row_format(true_table)
    pred_table_rows = column_to_row_format(pred_table)
    
    # 计算第二列F1
    second_column_f1 = calculate_second_column_f1(true_table_rows, pred_table_rows)
    
    # 计算表头F1
    table_header_f1 = calculate_table_header_f1(true_table_rows, pred_table_rows)
    
    # 计算数据单元格F1
    data_cell_f1 = calculate_data_cell_f1(true_table_rows, pred_table_rows)
    
    # 计算ERROR
    error_rate = calculate_error(true_table_rows, pred_table_rows)
    
    # 计算额外表头生成率
    extra_header_rate, extra_headers_with_data, true_headers = calculate_extra_header_rate(true_table_rows, pred_table_rows)
    
    # 整合结果
    results = {
        "id": sample.get("id", "unknown"),
        "second_column_f1": second_column_f1,
        "table_header_f1": table_header_f1,
        "data_cell_f1": data_cell_f1,
        "error_rate": error_rate,
        "extra_header_rate": extra_header_rate,
        "extra_header_details": {
            "extra_headers_count": len(extra_headers_with_data),
            "true_headers_count": len(true_headers),
            "extra_headers": list(extra_headers_with_data),
            "true_headers": list(true_headers)
        }
    }
    
    return results

def save_checkpoint(checkpoint_data, checkpoint_file):
    """保存检查点数据"""
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        logging.info(f"检查点已保存到: {checkpoint_file}")
    except Exception as e:
        logging.error(f"保存检查点失败: {e}")

def load_checkpoint(checkpoint_file):
    """加载检查点数据"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
    return None

def evaluate_sample_with_retry(sample, max_retries=MAX_RETRIES):
    """带重试机制的样本评估"""
    for attempt in range(max_retries):
        try:
            return evaluate_sample(sample)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"样本 {sample.get('id', 'unknown')} 评估失败: {e}")
                logging.error(traceback.format_exc())
                return {
                    "id": sample.get("id", "unknown"),
                    "error": str(e),
                    "second_column_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
                    "table_header_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
                    "data_cell_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
                    "error_rate": 0,
                    "extra_header_rate": 0,
                    "extra_header_details": {
                        "extra_headers_count": 0,
                        "true_headers_count": 0,
                        "extra_headers": [],
                        "true_headers": []
                    }
                }
            logging.warning(f"样本 {sample.get('id', 'unknown')} 第 {attempt + 1} 次评估失败，准备重试...")

def evaluate_metrics(refs, hyps, bleurt_scorer, bart_scorer, bert_scorer):
    # SacreBLEU
    bleu = BLEU().corpus_score(hyps, [refs]).score / 100

    # ROUGE-L
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = np.mean([rouge.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps)])

    # BLEURT
    bleurt_scores = bleurt_scorer.score(references=refs, candidates=hyps)
    bleurt_mean = np.mean(bleurt_scores)

    # BARTScore
    bart_scores = bart_scorer.score(hyps, refs, batch_size=4)
    bart_mean = np.mean(bart_scores)

    # BERTScore
    P, R, F1 = bert_scorer.score(hyps, refs)
    bert_p = P.mean().item()
    bert_r = R.mean().item()
    bert_f1 = F1.mean().item()

    # Content/Format P-Score/H-Score（示例：用BERTScore的P/R/F1代替，实际可自定义）
    content_p = bert_p
    format_p = bert_r
    content_h = bert_f1
    format_h = (bert_p + bert_r) / 2

    return {
        "SacreBLEU": bleu,
        "ROUGE-L": rouge_l,
        "BLEURT": bleurt_mean,
        "BARTScore": bart_mean,
        "Content P-Score": content_p,
        "Format P-Score": format_p,
        "Content H-Score": content_h,
        "Format H-Score": format_h,
    }

def main():
    """主函数，处理输入文件并输出评测结果"""
    # 使用直接在代码中指定的参数，而不是命令行参数
    input_file = INPUT_FILE
    model_name = MODEL_NAME
    batch_size = BATCH_SIZE
    
    # 初始化BERT Scorer模型
    init_bert_scorer()
    
    # 生成包含时间戳的输出文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取输入文件名（不含路径和扩展名）
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # 生成输出文件路径
    output_file = f"bert_metrics_{model_name}_{input_filename}_{timestamp}.json"
    checkpoint_file = f"checkpoint_{model_name}_{input_filename}_{timestamp}.json"
    
    # 创建结果目录（如果不存在）
    os.makedirs("eval_results", exist_ok=True)
    output_file = os.path.join("eval_results", output_file)
    checkpoint_file = os.path.join("eval_results", checkpoint_file)
    
    # 读取输入文件
    samples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logging.warning(f"无法解析行：{e}")
    except Exception as e:
        logging.error(f"无法读取输入文件：{e}")
        return
    
    logging.info(f"共读取 {len(samples)} 个样本进行评估")
    
    # 尝试加载检查点
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        logging.info("找到检查点，恢复评估进度...")
        results = checkpoint_data.get("results", [])
        processed_samples = checkpoint_data.get("processed_samples", 0)
        error_samples = checkpoint_data.get("error_samples", [])
    else:
        results = []
        processed_samples = 0
        error_samples = []
    
    # 评估每个样本
    aggregated_metrics = {
        "second_column_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
        "table_header_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
        "data_cell_f1": {metric: {"precision": 0, "recall": 0, "f1": 0} for metric in ['E', 'c', 'BS-scaled']},
        "error_rate": 0,
        "extra_header_rate": 0
    }
    
    # 创建进度条
    pbar = tqdm(total=len(samples), initial=processed_samples, desc="评估进度")
    
    try:
        for i in range(processed_samples, len(samples)):
            sample = samples[i]
            result = evaluate_sample_with_retry(sample)
            
            # 检查是否有错误
            if "error" in result:
                error_samples.append({
                    "id": result["id"],
                    "error": result["error"],
                    "sample": sample
                })
            else:
                results.append(result)
                
                # 累计结果用于计算平均值
                for component in ["second_column_f1", "table_header_f1", "data_cell_f1"]:
                    for metric in ['E', 'c', 'BS-scaled']:
                        for score in ["precision", "recall", "f1"]:
                            aggregated_metrics[component][metric][score] += result[component][metric][score]
                
                aggregated_metrics["error_rate"] += result["error_rate"]
                aggregated_metrics["extra_header_rate"] += result["extra_header_rate"]
            
            # 更新进度条
            pbar.update(1)
            
            # 定期保存检查点
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_data = {
                    "results": results,
                    "processed_samples": i + 1,
                    "error_samples": error_samples,
                    "aggregated_metrics": aggregated_metrics
                }
                save_checkpoint(checkpoint_data, checkpoint_file)
    
    except Exception as e:
        logging.error(f"评估过程中发生错误: {e}")
        logging.error(traceback.format_exc())
    finally:
        pbar.close()
    
    # 计算平均值
    num_valid_samples = len(results)
    if num_valid_samples > 0:
        for component in ["second_column_f1", "table_header_f1", "data_cell_f1"]:
            for metric in ['E', 'c', 'BS-scaled']:
                for score in ["precision", "recall", "f1"]:
                    aggregated_metrics[component][metric][score] /= num_valid_samples
        
        aggregated_metrics["error_rate"] /= num_valid_samples
        aggregated_metrics["extra_header_rate"] /= num_valid_samples
    
    # 输出汇总结果
    logging.info("\n===== 评估结果 =====")
    for component in ["second_column_f1", "table_header_f1", "data_cell_f1"]:
        logging.info(f"\n{component}:")
        for metric in ['E', 'c', 'BS-scaled']:
            logging.info(f"  {metric} 指标:")
            logging.info(f"    F1: {aggregated_metrics[component][metric]['f1']:.4f}")
            logging.info(f"    - Precision: {aggregated_metrics[component][metric]['precision']:.4f}")
            logging.info(f"    - Recall: {aggregated_metrics[component][metric]['recall']:.4f}")
    
    logging.info(f"\nERROR Rate: {aggregated_metrics['error_rate']:.4f}")
    logging.info(f"Extra Header Rate: {aggregated_metrics['extra_header_rate']:.4f}")
    
    if error_samples:
        logging.warning(f"\n评估过程中出现错误的样本数量: {len(error_samples)}")
        error_samples_file = os.path.join("eval_results", f"error_samples_{timestamp}.json")
        try:
            with open(error_samples_file, 'w', encoding='utf-8') as f:
                json.dump(error_samples, f, ensure_ascii=False, indent=2)
            logging.info(f"错误样本信息已保存到: {error_samples_file}")
        except Exception as e:
            logging.error(f"保存错误样本信息失败: {e}")
    
    # 保存结果到文件
    output_data = {
        "per_sample_results": results,
        "aggregated_metrics": aggregated_metrics,
        "error_samples": error_samples,
        "total_samples": len(samples),
        "valid_samples": num_valid_samples,
        "error_samples_count": len(error_samples)
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logging.info(f"\n结果已保存到: {output_file}")
    except Exception as e:
        logging.error(f"保存结果到文件失败: {e}")

if __name__ == "__main__":
    main() 
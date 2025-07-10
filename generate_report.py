import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import logging
import time
import yaml

#加载配置
def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
# 加载主配置

config = load_config()
llm_api_config = config['llm_api']
execution_config = config['execution']
file_paths_config = config['file_paths']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 从配置中读取参数
API_KEY = llm_api_config['api_key']# API密钥
BASE_URL = llm_api_config['base_url']# API基础URL
MODEL_NAME = llm_api_config['report_model'] # LLM模型名称
RETRY_ATTEMPTS = execution_config['retry_attempts'] # 重试次数
RETRY_DELAY = execution_config['retry_delay']# 重试延迟时间（秒）
INPUT_FILE = file_paths_config['step3_input']# 输入文件路径
FINAL_REPORT_FILE = file_paths_config['step3_output']# 最终报告文件路径

#LLM归纳功能
# 初始化OpenAI客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def summarize_needs_with_llm(topic: str, why_list: list, how_list: list) -> str:
    # 过滤掉列表中的无效项（空字符串、None、"未提及"）
    filtered_why = [item for item in why_list if item and item not in ["未提及", ""]]
    filtered_how = [item for item in how_list if item and item not in ["未提及", ""]]
    # 如果过滤后没有任何有效信息，则直接返回空字符串
    if not filtered_why and not filtered_how:
        return "" 
    # 将列表转换为带项目符号的字符串

    why_points = "\n".join([f"- {w}" for w in filtered_why])
    how_points = "\n".join([f"- {h}" for h in filtered_how])
    # 构建发送给大模型的Prompt

    prompt = f"""
你是一名资深的市场分析总监，你的任务是阅读关于“{topic}”的用户反馈，并提炼出核心的、简明的原始需求。

【用户核心痛点 - Why】:
{why_points if why_points else "无"}

【用户解决方案与建议 - How】:
{how_points if how_points else "无"}

请基于以上信息，总结出 1-3 条最核心、最普遍的用户原始需求。
要求：
- 每条需求都应是独立、完整、可行动的观点。
- 语言精炼，直指问题本质。
- 以无序列表的格式返回，每条需求占一行，以'- '开头。不要任何额外说明或标题。
"""
    # API调用重试循环
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # 调用LLM API
            response = client.chat.completions.create(
                model=MODEL_NAME,# 使用配置中的模型
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,# 较低的温度，使输出更具确定性和一致性
                max_tokens=500,# 限制生成长度
            )
            summary = response.choices[0].message.content
            return summary.strip()
        except Exception as e:
            logging.error(f"LLM归纳失败 (主题: {topic}, 尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "LLM归纳总结失败。"
    return "LLM归纳总结失败。"

#主处理逻辑
def main():
    logging.info(f"开始生成JSON分析报告，读取文件: {INPUT_FILE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"错误：输入文件未找到: '{INPUT_FILE}'")
        return
    except json.JSONDecodeError:
        logging.error(f"错误：文件内容不是有效的JSON格式: '{INPUT_FILE}'")
        return
    
    # 将嵌套的JSON数据转换为扁平的列表，便于用Pandas处理
    processed_data = []
    for row in all_data:# 遍历每条原始评论的分析结果
        analysis_data = row.get('analysis_results', {})# 获取分析结果
        if not analysis_data:
            continue
        for category, details in analysis_data.items():
            if isinstance(details, dict) and 'error' not in details:
                record = {
                    'sentiment': details.get('sentiment'),
                    'topic': details.get('topic'),
                    'why': details.get('primitive_need_4w1h', {}).get('why'),
                    'how': details.get('primitive_need_4w1h', {}).get('how'),
                }
                processed_data.append(record)
    
    if not processed_data:
        logging.warning("未能从输入文件中解析出任何有效的分析数据。")
        return
        
    df = pd.DataFrame(processed_data)
    logging.info(f"数据解析与展平完成，共处理 {len(df)} 条有效主题分析。")

    #统计分析 
    total_mentions = len(df) # 总提及次数（每个主题算一次）
    topic_counts = df['topic'].value_counts().reset_index() # 计算每个主题的提及次数
    topic_counts.columns = ['topic', 'mention_count'] # 重命名列
    # 计算每个主题的提及率
    topic_counts['mention_rate'] = (topic_counts['mention_count'] / total_mentions) * 100
    # 按主题和情感进行分组，计算每个情感的次数
    sentiment_distribution = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
    # 将提及数、提及率与情感分布合并到一个DataFrame中
    df_stats = pd.merge(topic_counts, sentiment_distribution, on='topic', how='left').fillna(0)
    # 按提及数降序排序
    df_stats = df_stats.sort_values(by='mention_count', ascending=False)

    logging.info("开始构建最终的JSON报告...")
    
    final_report = {
        "report_metadata": {
            "report_generated_at": pd.Timestamp.now().isoformat(),
            "source_file": INPUT_FILE,
            "total_topics_analyzed": total_mentions,
            "unique_topics_found": len(df_stats)
        },
        "analysis_by_topic": []
    }
    # 遍历每个主题，生成摘要并填充报告
    for _, row in tqdm(df_stats.iterrows(), total=len(df_stats), desc="生成报告内容"):
        topic = row['topic']
        
        topic_df = df[df['topic'] == topic]
        why_list = topic_df['why'].tolist()
        how_list = topic_df['how'].tolist()
        
        summary_text = summarize_needs_with_llm(topic, why_list, how_list)
        summary_list = [line.strip('- ').strip() for line in summary_text.split('\n') if line.strip()]

        sentiments = {
            "正面": int(row.get('正面', 0)),
            "负面": int(row.get('负面', 0)),
            "中性": int(row.get('中性', 0))
        }

        topic_entry = {
            "topic": topic,
            "statistics": {
                "mention_count": int(row['mention_count']),
                "mention_rate_percent": round(row['mention_rate'], 2),
                "sentiment_distribution": sentiments
            },
            "primitive_needs_summary": summary_list
        }
        
        final_report["analysis_by_topic"].append(topic_entry)

    with open(FINAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    logging.info(f"处理完成！完整的JSON报告已保存至: {FINAL_REPORT_FILE}")
    print(f"\n处理完成！\n最终报告文件: {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    main()
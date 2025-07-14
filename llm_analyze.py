import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import logging
import json
import concurrent.futures
import yaml

# --- 配置加载 ---
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_topics(topics_path):
    with open(topics_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
llm_api_config = config['llm_api']
execution_config = config['execution']
file_paths_config = config['file_paths']
data_columns_config = config['data_columns']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局常量 ---
API_KEY = llm_api_config['api_key']
BASE_URL = llm_api_config['base_url']
MODEL_NAME = llm_api_config['tagging_model']
MAX_WORKERS = execution_config['max_workers']
RETRY_ATTEMPTS = execution_config['retry_attempts']
RETRY_DELAY = execution_config['retry_delay']

INPUT_FILE = file_paths_config['step1_output'] 
OUTPUT_FILE = "output_analyzed.csv" 
TOPICS_FILE = file_paths_config['topics_config']
COMMENT_COLUMN = data_columns_config['comment_column_name']

TOPIC_LIBRARIES = load_topics(TOPICS_FILE)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 核心函数 ---
def analyze_sentiment_and_topic(comment: str, category: str) -> dict:
    """
    使用LLM分析单条评论的情感和细分主题。
    """
    if category not in TOPIC_LIBRARIES:
        return {"error": f"未找到'{category}'的细分主题库"}

    topics = TOPIC_LIBRARIES[category]
    formatted_topics = "\n- ".join(topics)

    prompt = f"""
你是一名汽车用户洞察分析师，任务是严谨地分析用户评论。

【用户评论】: "{comment}"
【分析维度】: "{category}"

请完成以下两项任务:

1.  情感判断: 用户对"{category}"的情感倾向是"正面"、"负面"还是"中性"？

2.  主题提炼: 从以下【候选主题】中，选择一个最能概括评论核心内容的主题。
    
    选择原则：
    - 优先选择最具体、最精确的主题
    - 如果评论提到具体配置（如"108度电池"、"激光雷达"、"空气悬挂"），选择对应的具体主题
    - 如果评论内容较宽泛，再选择较为概括的主题

【候选主题】:
- {formatted_topics}

请严格按照以下JSON格式返回结果，不要添加任何额外说明或注释，直接返回JSON对象本身：
{{
  "sentiment": "正面 或 负面 或 中性",
  "topic": "在此处填写最匹配的候选主题"
}}
"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                # response_format={"type": "json_object"}, # <--- 已移除此行
                max_tokens=200
            )
            result_str = response.choices[0].message.content
            analysis_result = json.loads(result_str)
            
            if 'sentiment' in analysis_result and 'topic' in analysis_result:
                return analysis_result
            else:
                raise ValueError("返回的JSON格式不完整")
        except Exception as e:
            logging.error(f"LLM分析失败 (类别: {category}, 尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": "LLM分析失败"}
    return {"error": "LLM分析失败"}

def process_row(row_tuple):
    """
    处理DataFrame的单行数据，为每个标签生成分析结果。
    """
    index, row = row_tuple
    comment = row[COMMENT_COLUMN]
    tags_str = row['标签']
    
    results = []
    
    if isinstance(tags_str, str) and tags_str != '未匹配到标签':
        try:
            categories_to_process = json.loads(tags_str.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            categories_to_process = [tag.strip() for tag in tags_str.split(',')]

        for category in categories_to_process:
            if category in TOPIC_LIBRARIES:
                analysis = analyze_sentiment_and_topic(comment, category)
                if "error" not in analysis:
                    results.append({
                        "comment": comment,
                        "category": category,
                        "topic": analysis.get("topic", "未知主题"),
                        "sentiment": analysis.get("sentiment", "未知情感")
                    })
    return results

# --- 主函数 ---
def main():
    logging.info(f"--- 步骤2: LLM轻量级分析 (情感与主题) ---")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except FileNotFoundError:
        logging.error(f"错误：输入文件未找到 '{INPUT_FILE}'")
        return

    all_analysis_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows_to_process = list(df.iterrows())
        future_to_row = {executor.submit(process_row, row): row for row in rows_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows_to_process), desc="分析评论中"):
            try:
                result_list = future.result()
                if result_list:
                    all_analysis_results.extend(result_list)
            except Exception as e:
                logging.error(f"处理某行时发生严重错误: {e}")

    if not all_analysis_results:
        logging.warning("分析完成，但未生成任何有效的分析结果。")
        return

    result_df = pd.DataFrame(all_analysis_results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    logging.info(f"--- 步骤2完成！分析结果已保存至: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()
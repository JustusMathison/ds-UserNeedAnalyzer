import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import logging
import json
import concurrent.futures
import yaml 

# 加载主配置
def load_config(config_path='config.yaml'):
    """加载主YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载细分主题库的函数
def load_topics(topics_path):
    """从指定的YAML文件加载细分主题库"""
    print(f"从 {topics_path} 加载细分主题库...")# 打印加载信息
    with open(topics_path, 'r', encoding='utf-8') as f:# 使用'utf-8'编码打开文件
        return yaml.safe_load(f)# 返回加载的内容

# 1. 加载所有配置
config = load_config()# 加载主配置文件
llm_api_config = config['llm_api']# 获取LLM API相关的配置
execution_config = config['execution']# 获取执行参数相关的配置
file_paths_config = config['file_paths']# 获取文件路径相关的配置

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. 从配置中读取运行参数
API_KEY = llm_api_config['api_key']# API密钥
BASE_URL = llm_api_config['base_url']# API基础URL
MODEL_NAME = llm_api_config['tagging_model'] # 选择的模型
MAX_WORKERS = execution_config['max_workers']# 最大并发线程数
RETRY_ATTEMPTS = execution_config['retry_attempts']# API调用失败的最大重试次数
RETRY_DELAY = execution_config['retry_delay']# 每次重试之间的延迟时间（秒）
INPUT_FILE = file_paths_config['step2_input']# 输入文件路径
OUTPUT_FILE = file_paths_config['step2_output']# 输出文件路径

# 3. 从配置文件加载细分主题库
SPECIFIC_TOPIC_LIBRARIES = load_topics(file_paths_config['topics_config'])
print("细分主题库加载成功！")

# 核心功能

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_analysis_for_category(comment: str, category: str) -> tuple:
    if category not in SPECIFIC_TOPIC_LIBRARIES:
        return category, {"error": "无对应主题库"}
    
    topics = SPECIFIC_TOPIC_LIBRARIES[category]
    formatted_topics = "\n- ".join(topics)
    
    prompt = f"""
你是一名顶级的汽车用户洞察分析师，任务是严谨地分析【用户评论】中关于“{category}”的内容。

【用户评论】:
"{comment}"

请一步步完成以下三项分析任务:
1.  情感判断: 用户对“{category}”的情感倾向是“正面”、“负面”还是“中性”？
2.  主题提炼: 从【候选主题】中，选择一个最能概括评论核心内容的具体主题。
3.  原始需求分析 (4W1H): 基于评论，尽可能提取用户需求的核心要素。如果评论中未明确提及，请将对应值留空或设为 "未提及"。
    - when (时间): 用户在什么场景或时机下会遇到这个问题或产生这个需求？
    - where (地点): 需求的发生地点在哪里？(例如：车内、车外、特定道路)
    - why (原因): 用户为什么会提出这个观点？背后的核心痛点或期望是什么？
    - who (人物): 这个需求/问题对哪类人群影响最大？(例如：驾驶员、儿童、家庭)
    - how (方式/建议): 用户是如何建议解决的，或他们期望如何使用一个理想的功能？

【候选主题】:
- {formatted_topics}

请严格按照以下JSON格式返回你的分析结果，不要添加任何额外说明或注释：
{{
  "sentiment": "正面 或 负面 或 中性",
  "topic": "在此处填写最匹配的候选主题",
  "primitive_need_4w1h": {{
    "when": "分析得出的时间或场景",
    "where": "分析得出的地点",
    "why": "分析出的核心原因/痛点",
    "who": "分析出的影响人群",
    "how": "分析出的解决方案/建议"
  }}
}}
"""
    for attempt in range(RETRY_ATTEMPTS):
        # 调用OpenAI兼容的聊天补全的接口
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,# 使用配置中的模型
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,# 设置温度为0.0，确保输出结果的确定性和一致性
                # 设置响应格式为JSON对象，确保返回的内容符合预期的结构
                response_format={"type": "json_object"},
                max_tokens=500,# 设置最大生成的token数
            )
            result_str = response.choices[0].message.content# 获取返回的字符串内容
            analysis_result = json.loads(result_str)# 将JSON字符串解析为Python字典
            # 验证返回的JSON是否包含所有必需的键
            if 'sentiment' in analysis_result and 'topic' in analysis_result and 'primitive_need_4w1h' in analysis_result:
                return category, analysis_result# 如果格式正确，返回类别和分析结果
            else:
                raise ValueError("返回的JSON格式不完整或结构错误")

        except Exception as e:
            # 记录API调用或解析失败的错误
            logging.error(f"API调用或解析失败 (类别: {category}, 尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return category, {"error": "分析失败"}
    return category, {"error": "分析失败"}


def process_comment(row_tuple):
    index, row = row_tuple # 解包元组
    comment = row['评论'] # 获取评论内容
    tags_str = row['标签'] # 获取标签字符串
    
    final_analysis = {}# 初始化一个空字典，用于存储该条评论所有标签的分析结果

    if isinstance(tags_str, str):# 检查标签是否为字符串
        # 筛选出那些在细分主题库中定义了的类别
        categories_to_process = [cat for cat in SPECIFIC_TOPIC_LIBRARIES.keys() if cat in tags_str]
        # 遍历需要处理的每个类别
        for category in categories_to_process:
            cat, analysis_result = get_analysis_for_category(comment, category)
            final_analysis[cat] = analysis_result
    # 返回一个结构化的字典，包含原始评论、标签和分析结果
    return {
        'original_comment': comment,
        'original_tags': tags_str,
        'analysis_results': final_analysis
    }

# 主脚本入口
def main():
    """主执行函数，封装了脚本的核心逻辑。"""
    # 设置日志记录格式和级别
    logging.info("脚本启动，将分析结果输出为JSON文件...")
    try:
        try:
            df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            logging.warning("UTF-8解码失败，尝试使用GBK编码...")
            df = pd.read_csv(INPUT_FILE, encoding='gbk')
    except FileNotFoundError:
        logging.error(f"错误：输入文件未找到: '{INPUT_FILE}'")
        return
    except Exception as e:
        logging.error(f"读取CSV文件时发生未知错误: {e}")
        return

    final_results = []# 初始化一个列表，用于存储所有评论的处理结果
    # 使用线程池实现并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_comment, row_tuple) for row_tuple in df.iterrows()]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="并发分析进度"):
            try:
                result_dict = future.result()
                final_results.append(result_dict)
            except Exception as e:
                logging.error(f"处理某行时发生严重错误: {e}")
    
    logging.info("--- 分析完成 ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    logging.info(f"结果已保存至JSON文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
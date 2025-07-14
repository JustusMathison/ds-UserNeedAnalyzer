import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import logging
import time
import yaml
from collections import Counter
import concurrent.futures

def load_config(config_path='config.yaml'):
    """加载配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# --- 配置加载 ---
config = load_config()
llm_api_config = config['llm_api']
execution_config = config['execution']
file_paths_config = config['file_paths']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局常量 ---
API_KEY = llm_api_config['api_key']
BASE_URL = llm_api_config['base_url']
MODEL_NAME = llm_api_config['report_model']
RETRY_ATTEMPTS = execution_config['retry_attempts']
RETRY_DELAY = execution_config['retry_delay']
MAX_WORKERS = execution_config['max_workers']
INPUT_FILE = "output_analyzed.csv"
FINAL_REPORT_FILE = file_paths_config['step3_output']

# --- 初始化OpenAI客户端 ---
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_aggregated_analysis_with_llm(topic: str, comments: list, sentiment: str) -> dict:
    """
    对一个主题下特定情感的所有评论进行聚合分析。
    (此函数与上一版本相同，无需修改)
    """
    formatted_comments = "\n".join([f"{i+1}. {c}" for i, c in enumerate(comments)])
    sentiment_instructions = {
        "负面": f"你的任务是基于以下关于【{topic}】的【负面】用户反馈集合，进行深度分析和总结。请聚焦于用户的核心痛点、不满、抱怨的具体场景和对解决方案的期望。",
        "正面": f"你的任务是基于以下关于【{topic}】的【正面】用户反馈集合，进行深度分析和总结。请聚焦于用户的核心赞扬点、满意之处、喜爱原因以及他们认为产品做对了什么。",
        "中性": f"你的任务是基于以下关于【{topic}】的【中性】用户反馈集合，进行客观地分析和总结。请描述用户陈述的事实、提出的问题或中立的观察点。"
    }
    task_description = sentiment_instructions.get(sentiment, f"你的任务是基于以下关于【{topic}】的用户反馈集合，进行深度分析和总结。")
    prompt = f"""
你是一名顶级的汽车市场洞察总监。{task_description}

【用户反馈集合】:
{formatted_comments}

请一步步完成以下分析任务:
1.  **4W1H 总结**: 综合所有反馈，总结出该情感倾向下的共性需求场景要素。如果某些方面信息不足，请客观地填写“信息不足”。
    -   When (时机/场景): 用户在什么典型场景下会表达出对“{topic}”的这种({sentiment})看法？
    -   Where (地点): 需求的发生地点在哪里？(例如：高速公路、市区拥堵、停车场、车内)
    -   Who (人物): 这个问题对哪类人群影响最大？(例如：驾驶员、家庭成员、新手司机、注重体验的用户)
    -   Why (核心原因): 用户产生这种({sentiment})看法的根本原因、核心痛点或核心期望是什么？
    -   How (解决方案/建议): 用户提出了哪些具体的解决方案，或者他们期望的理想功能是怎样的？

2.  **核心需求提炼**: 基于以上所有信息，总结出1-3条最核心、最普遍的用户原始需求。每条需求都应是独立、完整、可行动的观点。

请严格按照以下JSON格式返回你的分析报告，确保key和结构完全一致，不要添加任何额外说明或注释，直接返回JSON对象本身：
{{
  "consolidated_4w1h": {{
    "when": "总结出的共性时机或场景",
    "where": "总结出的共性地点",
    "why": "总结出的核心原因/痛点",
    "who": "总结出的核心影响人群",
    "how": "总结出的共性解决方案/建议"
  }},
  "primitive_needs_summary": [
    "提炼出的第一条核心原始需求",
    "提炼出的第二条核心原始需求"
  ]
}}
"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            result_str = response.choices[0].message.content
            analysis_result = json.loads(result_str)
            return analysis_result
        except Exception as e:
            logging.error(f"LLM聚合分析失败 (主题: {topic}, 情感: {sentiment}, 尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return {
        "consolidated_4w1h": {"error": f"LLM分析在{RETRY_ATTEMPTS}次尝试后失败"},
        "primitive_needs_summary": ["LLM分析失败"]
    }

def process_topic_group(group_info):
    """
    并发处理单个主题分组。
    (此函数与上一版本相同，无需修改)
    """
    (category, topic), group = group_info
    total_mention_count = len(group)
    sentiment_counts = Counter(group['sentiment'])
    overall_statistics = {
        "mention_count": total_mention_count,
        "sentiment_distribution": {
            "正面": sentiment_counts.get("正面", 0),
            "负面": sentiment_counts.get("负面", 0),
            "中性": sentiment_counts.get("中性", 0)
        }
    }
    analysis_by_sentiment = []
    sentiment_order = ["负面", "正面", "中性"]
    for sentiment in sentiment_order:
        sentiment_group = group[group['sentiment'] == sentiment]
        if sentiment_group.empty:
            continue
        comments_for_sentiment = sentiment_group['comment'].tolist()
        aggregated_analysis = get_aggregated_analysis_with_llm(topic, comments_for_sentiment, sentiment)
        sentiment_entry = {
            "sentiment": sentiment,
            "mention_count": len(sentiment_group),
            "consolidated_analysis": aggregated_analysis.get("consolidated_4w1h", {}),
            "core_needs": aggregated_analysis.get("primitive_needs_summary", []),
            "example_comments": sentiment_group['comment'].head(3).tolist()
        }
        analysis_by_sentiment.append(sentiment_entry)
    topic_entry = {
        "topic_name": topic,
        "statistics": overall_statistics,
        "analysis_by_sentiment": analysis_by_sentiment
    }
    return category, topic_entry

def main():
    """
    【重构】主函数，将最终报告调整为按全局提及量排序的扁平列表。
    """
    logging.info(f"--- 步骤3: 按情感聚合分析与报告生成 ---")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except FileNotFoundError:
        logging.error(f"错误：输入文件未找到 '{INPUT_FILE}'")
        return

    grouped = list(df.groupby(['category', 'topic']))

    # 【改动】不再使用字典，而是用一个列表来存放所有话题的分析结果
    all_topics_analysis = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_group = {executor.submit(process_topic_group, group_info): group_info for group_info in grouped}
        pbar = tqdm(concurrent.futures.as_completed(future_to_group), total=len(grouped), desc="聚合分析中")
        for future in pbar:
            try:
                # process_topic_group 返回 (category, topic_entry)
                category, topic_entry = future.result()
                pbar.set_description(f"处理完成: {topic_entry['topic_name']}")

                # 【改动】将 category 作为一个字段添加到 topic_entry 内部
                topic_entry['category'] = category

                # 【改动】将处理好的 topic_entry 添加到最终的扁平列表中
                all_topics_analysis.append(topic_entry)
            except Exception as e:
                group_info = future_to_group[future]
                category, topic = group_info[0]
                logging.error(f"处理主题'{topic}'(类别:{category})时发生严重错误: {e}")

    # 【改动】对整个列表按总提及量进行全局排序
    all_topics_analysis.sort(key=lambda x: x['statistics']['mention_count'], reverse=True)

    # 【改动】构建新的、扁平化的输出结构
    output_data = {
        "report_metadata": {
            "report_generated_at": pd.Timestamp.now().isoformat(),
            "source_file": INPUT_FILE,
            "total_topics_analyzed": len(grouped),
            "total_mentions_analyzed": len(df)
        },
        "all_topics_analysis": all_topics_analysis  # 使用新的排序后列表
    }

    with open(FINAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logging.info(f"--- 步骤3完成！最终报告已保存至: {FINAL_REPORT_FILE} ---")
    print(f"\n处理完成！\n最终报告文件: {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    main()
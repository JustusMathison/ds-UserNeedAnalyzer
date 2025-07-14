

import time
import tag_process      # 步骤1: 关键词打标
import llm_analyze      # 步骤2: 轻量级LLM分析 (情感+主题)
import generate_report  # 步骤3: 聚合LLM分析与报告生成

def run_full_pipeline():
    """
    按顺序执行优化后的数据分析流程。
    """
    start_time = time.time()
    print("=========================================")
    print("🚀 开始执行优化版数据分析流程...")
    print("=========================================\n")

    # 步骤 1: 关键词初步打标
    print("--- [步骤 1/3] 开始执行关键词初步打标... ---")
    try:
        tag_process.main()
        print("--- [步骤 1/3] 初步打标完成！ ---\n")
    except Exception as e:
        print(f"❌ 步骤 1 执行失败: {e}")
        return

    # 步骤 2: LLM轻量级分析（情感 & 主题）
    print("--- [步骤 2/3] 开始执行 LLM 轻量级分析 (情感与主题)... ---")
    print("（此步骤将为每条相关评论调用一次LLM，请耐心等待...）")
    try:
        llm_analyze.main()
        print("--- [步骤 2/3] LLM 轻量级分析完成！ ---\n")
    except Exception as e:
        print(f"❌ 步骤 2 执行失败: {e}")
        return

    # 步骤 3: 聚合分析与报告生成
    print("--- [步骤 3/3] 开始聚合分析并生成最终报告... ---")
    print("（此步骤将为每个主题调用一次LLM进行深度总结，效率更高...）")
    try:
        generate_report.main()
        print("--- [步骤 3/3] 最终报告生成完成！ ---\n")
    except Exception as e:
        print(f"❌ 步骤 3 执行失败: {e}")
        return

    end_time = time.time()
    total_time = end_time - start_time
    print("=========================================")
    print(f"✅ 全部分析流程执行完毕！总耗时: {total_time:.2f} 秒。")
    print("=========================================")

if __name__ == "__main__":
    run_full_pipeline()
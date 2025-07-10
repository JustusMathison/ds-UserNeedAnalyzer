# main.py
import time
import tag_process  # 导入第一个脚本
import llm_tag      # 导入第二个脚本
import generate_report # 导入第三个脚本

def run_full_pipeline():
    """
    按顺序执行整个数据分析流程。
    """
    start_time = time.time()
    
    print("=========================================")
    print("🚀 开始执行完整数据分析流程...")
    print("=========================================\n")

    # --- 步骤 1: 关键词初步打标 ---
    print("--- [步骤 1/3] 开始执行关键词初步打标... ---")
    try:
        tag_process.main()
        print("--- [步骤 1/3] 初步打标完成！ ---\n")
    except Exception as e:
        print(f"❌ 步骤 1 执行失败: {e}")
        return # 如果失败则中断流程

    # --- 步骤 2: LLM 深度分析 ---
    print("--- [步骤 2/3] 开始执行 LLM 深度分析... ---")
    print("（此步骤耗时较长，请耐心等待...）")
    try:
        llm_tag.main()
        print("--- [步骤 2/3] LLM 深度分析完成！ ---\n")
    except Exception as e:
        print(f"❌ 步骤 2 执行失败: {e}")
        return

    # --- 步骤 3: 生成最终分析报告 ---
    print("--- [步骤 3/3] 开始生成最终分析报告... ---")
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
import pandas as pd
import re
import yaml 

def load_config(config_path='config.yaml'):
   # 加载主YAML配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_keywords(keywords_path):
    # 从指定的YAML文件加载关键词库
    print(f"从 {keywords_path} 加载关键词库...")
    with open(keywords_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def tag_comment(comment, rules):
    
    # 根据规则为单条评论打上所有匹配的标签
    
    if not isinstance(comment, str):# 检查输入是否为字符串，如果不是（例如是空值NaN），则直接返回空列表
        return []
        
    found_tags = []# 初始化一个空列表，用于存放找到的标签
    for tag_name, pattern in rules.items():# 遍历规则字典中的每一个标签和对应的正则模式
        if re.search(pattern, comment, re.IGNORECASE):# 使用re.search进行不区分大小写的匹配
            found_tags.append(tag_name)# 如果匹配成功，将标签名添加到列表中
    return found_tags# 返回包含所有匹配标签的列表

def process_csv_file(input_path, output_path, column_name, tagging_rules): 
    """
    读取CSV文件，进行打标，并保存到新文件
    """
    try:
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='gbk')
        print(f"成功读取文件: {input_path}")
    except FileNotFoundError:
        print(f"错误：输入文件未找到，请检查路径 '{input_path}' 是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return

    if column_name not in df.columns:
        print(f"错误：文件中未找到指定的列 '{column_name}'。请检查列名是否正确。")
        print(f"文件包含的列有: {list(df.columns)}")
        return

    df['标签'] = df[column_name].apply(lambda x: tag_comment(x, tagging_rules))
    df.loc[df['标签'].str.len() == 0, '标签'] = '未匹配到标签'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"处理完成！打标后的文件已保存至: {output_path}")

def main():
    """
    主执行函数，封装了脚本的核心逻辑。
    """
    config = load_config()# 加载主配置文件
    file_paths = config['file_paths']# 获取文件路径配置
    data_columns = config['data_columns']# 获取数据列配置
    #加载关键词库
    KEYWORDS = load_keywords(file_paths['keywords_config'])# 加载关键词配置文件
    print("关键词库加载成功！")
    #构建正则表达式规则
    TAGGING_RULES = {tag: '|'.join(keywords) for tag, keywords in KEYWORDS.items()}
    TAGGING_RULES['价格'] += r'|(\d{1,3}(\.\d{1,2})?)\s*[万wW]'# 匹配如 "15万", "20.5w" 等价格表述
    TAGGING_RULES['能耗'] += r'|(\d{3,4})\s*(km|公里|续航)'# 匹配如 "600km", "700公里" 等能耗/续航表述
    TAGGING_RULES['动力'] += r'|(零百|0-100)\D*(\d\.?\d*)\s*秒'# 匹配如 "零百3.8秒", "0-100加速5s" 等动力表述
    # 获取输入输出文件路径和列名
    input_file = file_paths['step1_input']# 输入文件名
    output_file = file_paths['step1_output']# 输出文件名
    comment_column = data_columns['comment_column_name'] # 评论列名
    
    process_csv_file(input_file, output_file, column_name=comment_column, tagging_rules=TAGGING_RULES)

if __name__ == "__main__":
    main()
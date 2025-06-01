import os
import re

def load_imdb_data(data_dir):
    """
    加载 IMDB 电影评论数据集。
    数据集结构预期为 data_dir/pos/*.txt 和 data_dir/dir/*.txt

    Args:
        data_dir (str): 包含 'pos' 和 'neg' 子目录的数据集路径。

    Returns:
        list: 包含 (label, text) 元组的列表。
              label: 1 for positive, 0 for negative.
              text: 评论文本。
    """
    data = []
    
    # 加载正面评论 (label = 1)
    pos_dir = os.path.join(data_dir, 'pos')
    if os.path.exists(pos_dir):
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(pos_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append((1, text)) # 1 代表正面
    else:
        print(f"Warning: Positive reviews directory not found: {pos_dir}")

    # 加载负面评论 (label = 0)
    neg_dir = os.path.join(data_dir, 'neg')
    if os.path.exists(neg_dir):
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(neg_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append((0, text)) # 0 代表负面
    else:
        print(f"Warning: Negative reviews directory not found: {neg_dir}")
        
    return data

def clean_text(text):
    """
    对原始 IMDB 评论进行基本清洗：
    - 移除 HTML 标签
    - 移除标点符号 (保留英文、数字和空格)
    - 转换为小写
    """
    # 移除 HTML 标签
    text = re.sub(r'<br />', ' ', text) # 将 <br /> 替换为空格
    text = re.sub(r'<.*?>', '', text) # 移除所有其他HTML标签 (通用性更强)

    # 移除特殊字符和标点符号，只保留字母、数字和空格
    # re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    # ^ 表示非，即匹配所有不是字母、数字或空格的字符
    # 将匹配到的字符替换为空字符串
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    
    # 转换为小写
    text = text.lower()
    
    # 移除多余的空格，并strip()移除字符串两端的空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
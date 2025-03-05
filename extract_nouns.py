# import spacy
# import json
# from tqdm import tqdm
# import torch

# # 检查是否有可用的GPU
# if torch.cuda.is_available():
#     spacy.require_gpu()
#     print("Using GPU")
# else:
#     print("GPU not available, using CPU")

# # 加载 GPU 版本的 SpaCy 模型
# nlp = spacy.load("en_core_web_sm")

# # 需要排除的词汇
# excluded_words = {"people", "person"}

# def extract_nouns(captions):
#     results = []
#     for item in captions:
#         sen_id = item['sen_id']
#         caption = item['caption']
#         doc = nlp(caption)
#         # 提取名词并排除特定的词
#         nouns = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in excluded_words]
#         results.append({'sen_id': sen_id, 'nouns': nouns})
#     return results

# def process_json(input_file, output_file, batch_size=10):
#     # 读取 JSON 文件
#     with open(input_file, 'r') as infile:
#         data = json.load(infile)
    
#     total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
#     processed_data = []

#     # 处理数据并显示进度条
#     for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=total_batches):
#         batch = data[i:i + batch_size]
#         processed_batch = extract_nouns(batch)
#         processed_data.extend(processed_batch)

#     # 写入新的 JSON 文件
#     with open(output_file, 'w') as outfile:
#         json.dump(processed_data, outfile, indent=4)

# # 设置输入和输出文件路径
# input_file_path = '/rds/projects/w/wanjz-text-video-retrieval/MSRVTT/videos/sentences.json' 
# output_file_path = '/rds/projects/w/wanjz-text-video-retrieval/MSRVTT/videos/nouns.json' 

# # 处理文件
# process_json(input_file_path, output_file_path, batch_size=10)

# print(f"名词信息已保存到 {output_file_path}")


import spacy
import json
from tqdm import tqdm

# 加载 SpaCy 模型（CPU 版本）
nlp = spacy.load("en_core_web_sm")

# 需要排除的词汇
excluded_words = {"people", "person", "man", "woman", "child"}

def extract_nouns(captions):
    results = []
    for item in captions:
        sen_id = item['sen_id']
        caption = item['caption']
        doc = nlp(caption)
        # 提取名词并排除特定的词
        nouns = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in excluded_words]
        results.append({'sen_id': sen_id, 'nouns': nouns})
    return results

def process_json(input_file, output_file, batch_size=10):
    # 读取 JSON 文件
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    processed_data = []

    # 处理数据并显示进度条
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=total_batches):
        batch = data[i:i + batch_size]
        processed_batch = extract_nouns(batch)
        processed_data.extend(processed_batch)

    # 写入新的 JSON 文件
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)

# 设置输入和输出文件路径
input_file_path = '/rds/projects/w/wanjz-text-video-retrieval/MSRVTT/videos/sentences.json' 
output_file_path = '/rds/projects/w/wanjz-text-video-retrieval/MSRVTT/videos/nouns.json' 


# 处理文件
process_json(input_file_path, output_file_path, batch_size=10)

print(f"名词信息已保存到 {output_file_path}")


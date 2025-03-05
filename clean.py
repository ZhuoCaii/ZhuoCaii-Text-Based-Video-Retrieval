# Replaice caption
import json

# Step 1: 读取原始 JSON 文件
input_file = 'MSRVTT/videos/combined_wordnet.json'  # 替换为你的输入文件名
output_file = 'MSRVTT/videos/combined_wordnet_nounsonly_input.json'

# 读取原始 JSON 数据
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: 处理数据
# 创建一个新的列表来保存处理后的句子数据
processed_sentences = []

# 遍历原始数据中的 sentences 列表
for sentence in data.get('sentences', []):
    # 创建一个新的字典，不包括原有的 caption 字段
    new_sentence = {
        'caption': sentence.get('explanation'),  # 将 explanation 字段重命名为 caption
        'video_id': sentence.get('video_id'),
        'sen_id': sentence.get('sen_id')
    }
    # 将处理后的句子添加到新列表中
    processed_sentences.append(new_sentence)

# 更新数据中的 sentences 列表
data['sentences'] = processed_sentences

# Step 3: 保存到新 JSON 文件
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"Processed data has been saved to {output_file}")

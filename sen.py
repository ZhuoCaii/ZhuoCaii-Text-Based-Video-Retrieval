import json

path_input='MSRVTT/videos/MSRVTT_data.json'
path_output='MSRVTT/videos/sentences.json'
# 加载JSON数据
with open(path_input, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取'sentences'部分
sentences = data.get('sentences', [])

# 提取sen_id和caption
extracted_data = [{'sen_id': sentence['sen_id'], 'caption': sentence['caption']} for sentence in sentences]

# 如果你想将结果保存到一个新的JSON文件
with open(path_output, 'w', encoding='utf-8') as outfile:
    json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)


# /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python MSRVTT/videos/count.py

# import json
# def count_video_occurrences(file_path):
#     count = 0
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             count += line.count('"video')
#     return count

# file_path = 'MSRVTT/videos/yolo_results.json'
# video_count = count_video_occurrences(file_path)
# print(f"Total number of 'video' entries: {video_count}")

# import cv2

# def get_video_frame_size(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video file: {video_path}")
#         return None

#     # 获取视频的帧宽和帧高
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     cap.release()
#     return frame_width, frame_height

# video_path = 'MSRVTT/videos/all/video3.mp4'
# size = get_video_frame_size(video_path)
# if size:
#     print(f"Video frame size: {size[0]}x{size[1]}")
# else:
#     print("Unable to determine video frame size.")

import json
from collections import Counter

# 读取原始 JSON 数据
input_file = 'MSRVTT/videos/gpt2_context_input.json'  # 替换为你的输入文件名

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取 sentences 列表
sentences = data.get('sentences', [])

# 计算每个句子的词数
word_counts = [len(sentence.get('caption', '').split()) for sentence in sentences]

# 统计词数的分布
word_count_distribution = Counter(word_counts)

# 打印统计信息
print("Word count distribution:", word_count_distribution)
print("Maximum word count:", max(word_counts))
print("Minimum word count:", min(word_counts))
print("Average word count:", sum(word_counts) / len(word_counts))

import os
import pandas as pd
import json


video_dir = 'MSRVTT/videos/all'
train_csv_path = 'MSRVTT/videos/MSRVTT_train.9k.csv'
test_csv_path = 'MSRVTT/videos/MSRVTT_JSFUSION_test.csv'
json_file_path = 'MSRVTT/videos/MSRVTT_data.json'

# def delete_even_videos(video_dir):
#     # 获取所有要删除的视频文件名
#     video_files = [file for file in os.listdir(video_dir) if file.lower().endswith('.mp4')]
#     video_ids_to_delete = []
    
#     for file_name in video_files:
#         # 提取视频ID
#         base_name = os.path.splitext(file_name)[0]
#         try:
#             video_id = int(base_name.replace('video', ''))
#             if video_id % 2 == 0:  # 检查视频ID是否为偶数
#                 video_ids_to_delete.append(base_name)
#                 os.remove(os.path.join(video_dir, file_name))
#                 print(f"Deleted {file_name}")
#         except ValueError:
#             continue  # 跳过无法转换为整数的文件
    
#     return video_ids_to_delete


# def delete_even_video_ids_from_csv(csv_path):
#     # 读取 CSV 文件
#     df = pd.read_csv(csv_path)
#     print(f"Original number of rows in {csv_path}: {len(df)}")
    
#     if 'video_id' in df.columns:
#         # 转换 'video_id' 列为字符串，然后提取数字并删除偶数 ID
#         df['video_id'] = df['video_id'].astype(str)
#         df['video_id_num'] = df['video_id'].str.replace('video', '').astype(int)
#         # 保留只有 ID 为奇数的行
#         df_filtered = df[df['video_id_num'] % 2 != 0]
#         df_filtered = df_filtered.drop(columns=['video_id_num'])  # 删除辅助列
        
#         # 将更新后的 DataFrame 保存回 CSV
#         df_filtered.to_csv(csv_path, index=False)
#         print(f"Updated {csv_path}. Number of rows after update: {len(df_filtered)}")
#     else:
#         print(f"Error: 'video_id' column not found in {csv_path}")




def update_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    def is_odd_video(video_id):
        video_num = int(''.join(filter(str.isdigit, video_id)))
        return video_num % 2 != 0

    # videos and sentences are list
    data['videos'] = [video for video in data['videos'] if is_odd_video(video['video_id'])]
    data['sentences'] = [sentence for sentence in data['sentences'] if is_odd_video(sentence['video_id'])]

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"已成功删除偶数 video_id 相关的数据，并将结果保存回原来的文件：{json_file_path}")




def main():
    
#     delete_even_video_ids_from_csv(train_csv_path)
#     delete_even_video_ids_from_csv(test_csv_path)

#     video_ids_to_delete = delete_even_videos(video_dir)
    update_json_file(json_file_path)

if __name__ == '__main__':
    main()

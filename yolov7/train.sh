# Do data preprocessing
module purge
module load bear-apps/2022a
module load spaCy/3.4.4-foss-2022a
module load Transformers/4.24.0-foss-2022a-CUDA-11.7.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load NLTK/3.8-foss-2022a

cd /rds/projects/w/wanjz-text-video-retrieval
/rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python CLIP4Clip/gpt2_context.py

cd /rds/projects/w/wanjz-text-video-retrieval
/rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python CLIP4Clip/gpt2_nocontext.py

cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/gpt2_context.py > gpt2_context.log 2>&1 &


cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/gpt2_nocontext.py > gpt2_nocontext.log 2>&1 &


cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/yolo_nocontext.py > yolo_nocontext.log 2>&1 &


cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/combined_gpt2_nocontext.py > combined_gpt2_nocontext.log 2>&1 &

 
cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/combined_gpt2_context.py > combined_gpt2_context.log 2>&1 &


cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/combined_gpt2.py > combined_gpt2.log 2>&1 &


cd /rds/projects/w/wanjz-text-video-retrieval
/rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/combined_wordnet.py

cd /rds/projects/w/wanjz-text-video-retrieval
/rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/wordnet_spacy.py

cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/combined_wordnet.py > combined_wordnet.log 2>&1 &


nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python /rds/projects/w/wanjz-text-video-retrieval/CLIP4Clip/gpt2_context.py >> gpt2_context.log 2>&1 &



#Download YOLOv7 model 
wget -P yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

module load bear-apps/2022a
module load torchvision/0.15.2-foss-2022a-CUDA-11.7.0
module load OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
module load matplotlib/3.5.2-foss-2022a
module load Seaborn/0.12.1-foss-2022a

cd /rds/projects/w/wanjz-text-video-retrieval
nohup /rds/bear-apps/2022a/EL8-ice/software/Python/3.10.4-GCCcore-11.3.0/bin/python yolov7/yolo_det.py > yolo_detection.log 2>&1 &


# Check PID
lsof | grep gpt2_nocontext.log
lsof | grep gpt2_context.log
lsof | grep yolo_nocontext.log
lsof | grep combined_gpt2.log
lsof | grep wordnet.log
lsof | grep combined_gpt2_nocontext.log
lsof | grep benchmark.log
lsof | grep wordnet_spacy.log
lsof | grep combined_wordnet_nounsonly.log
lsof | grep nohup.out

dmesg | grep -i 'oom'


# Train models
module purge
module load bear-apps/2022b
module load CUDA/12.0.0
module load PyTorch/2.1.2-foss-2022b-CUDA-12.0.0
module load torchvision/0.16.0-foss-2022b-CUDA-12.0.0

which python

# Experiment: SpaCy+GPT2+Append
cd /rds/projects/w/wanjz-text-video-retrieval
export DATA_PATH=MSRVTT/videos
nohup /rds/bear-apps/2022b/EL8-ice/software/Python/3.10.8-GCCcore-12.2.0/bin/python CLIP4Clip/main_task_retrieval.py --do_train \
--num_thread_reader=7 \
--epochs=13 \
--batch_size=32 \
--n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/gpt2_nocontext_input.json \
--features_path ${DATA_PATH}/all \
--output_dir CLIP4Clip/ckpts/gpt2_nocontext \
--lr 2.5e-5 \
--max_words 77 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
>> gpt2_nocontext.log 2>&1 &

# Experiment: SpaCy+YOLO+WordNet+Append
cd /rds/projects/w/wanjz-text-video-retrieval
export DATA_PATH=MSRVTT/videos
nohup /rds/bear-apps/2022b/EL8-ice/software/Python/3.10.8-GCCcore-12.2.0/bin/python CLIP4Clip/main_task_retrieval.py \
--do_train \
--num_thread_reader 7 \
--epochs 13 \
--batch_size 32 \
--n_display 50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/combined_wordnet_input.json \
--features_path ${DATA_PATH}/all \
--output_dir CLIP4Clip/ckpts/wordnet \
--lr 2.5e-5 \
--max_words 77 \
--max_frames 12 \
--batch_size_val 16 \
--datatype=msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
>> wordnet.log 2>&1 &


# Experiment: SpaCy+GPT2+Context+Append
cd /rds/projects/w/wanjz-text-video-retrieval
export DATA_PATH=MSRVTT/videos
nohup /rds/bear-apps/2022b/EL8-ice/software/Python/3.10.8-GCCcore-12.2.0/bin/python CLIP4Clip/main_task_retrieval.py \
--do_train \
--num_thread_reader 7 \
--epochs 13 \
--batch_size 32 \
--n_display 50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/gpt2_context_input.json \
--features_path ${DATA_PATH}/all \
--output_dir CLIP4Clip/ckpts/gpt2_context \
--lr 2.5e-5 \
--max_words 77 \
--max_frames 12 \
--batch_size_val 16 \
--datatype=msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
>> gpt2_context.log 2>&1 &

# Experiment: SpaCy+WordNet+Append
cd /rds/projects/w/wanjz-text-video-retrieval
export DATA_PATH=MSRVTT/videos
nohup /rds/bear-apps/2022b/EL8-ice/software/Python/3.10.8-GCCcore-12.2.0/bin/python CLIP4Clip/main_task_retrieval.py \
--do_train \
--num_thread_reader 7 \
--epochs 13 \
--batch_size 32 \
--n_display 50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/wordnet_spacy_input.json \
--features_path ${DATA_PATH}/all \
--output_dir CLIP4Clip/ckpts/wordnet_spacy \
--lr 2.5e-5 \
--max_words 77 \
--max_frames 12 \
--batch_size_val 16 \
--datatype=msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
>> wordnet_spacy.log 2>&1 &

# Experiment: SpaCy+YOLO+WordNet+Replace
cd /rds/projects/w/wanjz-text-video-retrieval
export DATA_PATH=MSRVTT/videos
nohup /rds/bear-apps/2022b/EL8-ice/software/Python/3.10.8-GCCcore-12.2.0/bin/python CLIP4Clip/main_task_retrieval.py \
--do_train \
--num_thread_reader 7 \
--epochs 30 \
--batch_size 32 \
--n_display 50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/combined_wordnet_nounsonly_input.json \
--features_path ${DATA_PATH}/all \
--output_dir CLIP4Clip/ckpts/combined_wordnet_nounsonly \
--lr 2.5e-5 \
--max_words 77 \
--max_frames 12 \
--batch_size_val 16 \
--datatype=msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
>> combined_wordnet_nounsonly.log 2>&1 &



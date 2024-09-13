import os
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from tqdm import tqdm
import logging

def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path, map_location=device)
    model.to(device).eval()
    return model, device

def detect_objects(video_path, model, device):
    video_path = Path(video_path)
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return []

    all_detections = set()
    imgsz = 640

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=imgsz, mode='bilinear', align_corners=False)

        with torch.no_grad():
            pred = model(img.to(device))[0]
            pred = non_max_suppression(pred, 0.4, 0.5)
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = f'{model.names[int(cls)]}'
                        all_detections.add(label)

    cap.release()
    unique_detections = set(all_detections)
    return list(unique_detections)

def process_batch(video_files, model, device, output_json_path, batch_size):
    for i in range(0, len(video_files), batch_size):
        batch_files = video_files[i:i + batch_size]
        batch_results = {}

        for video_file in batch_files:
            video_id = video_file.stem
            logging.info(f"Processing video: {video_id}")
            try:
                detected_objects = detect_objects(video_file, model, device)
                batch_results[video_id] = detected_objects
                logging.info(f"Detected objects in {video_id}: {detected_objects}")
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue

        temp_file_path = output_json_path + '.tmp'
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as json_file:
                existing_results = json.load(json_file)
        else:
            existing_results = {}

        existing_results.update(batch_results)

        with open(temp_file_path, 'w') as temp_file:
            json.dump(existing_results, temp_file, indent=4)

        os.rename(temp_file_path, output_json_path)
        logging.info(f"Batch {i // batch_size + 1} processed and results saved to {output_json_path}")

def main():
    weights_path = 'yolov7/yolov7-tiny.pt'
    video_folder_path = 'MSRVTT/videos/all'
    output_json_path = 'MSRVTT/videos/yolo_results.json'
    batch_size = 100

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model, device = load_model(weights_path)
    video_folder = Path(video_folder_path)
    video_files = list(video_folder.glob('video*.mp4'))
    
    process_batch(video_files, model, device, output_json_path, batch_size)

    logging.info("All are finished successfully!")
if __name__ == '__main__':
    main()



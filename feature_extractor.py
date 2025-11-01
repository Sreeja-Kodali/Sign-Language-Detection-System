import os
import json
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

DATASET_DIR = "dataset"
JSON_FILE = os.path.join(DATASET_DIR, "nslt_100.json")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
OUTPUT_FILE = "data/gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def extract_landmarks(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]
    return None

print(f"Loading metadata from {JSON_FILE}")
with open(JSON_FILE, "r") as f:
    metadata = json.load(f)

data = []
for vid_id, info in tqdm(metadata.items(), desc="Processing videos"):
    video_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
    if not os.path.exists(video_path):
        continue
    label = info["action"][0]
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frames // 5)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            landmarks = extract_landmarks(frame)
            if landmarks:
                data.append([label] + landmarks)
        count += 1
    cap.release()

os.makedirs("data", exist_ok=True)
cols = ["label"] + [f"f{i}" for i in range(len(data[0]) - 1)]
pd.DataFrame(data, columns=cols).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Features saved to {OUTPUT_FILE}")

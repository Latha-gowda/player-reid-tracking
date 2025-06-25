import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
import os
import torch

yolo_model = YOLO("best.pt")

# Load ResNet50 for player embedding extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tracker = DeepSort(max_age=30)

def extract_features(image):
    try:
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = resnet(image_tensor)
        return features.squeeze().numpy()
    except:
        return np.zeros(2048)  

def detect_and_track(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]  
        if results.boxes is None:
            frame_id += 1
            continue

        detections = results.boxes.data.cpu().numpy()  

        bboxes = []
        for box in detections:
            x1, y1, x2, y2, conf, cls = box[:6]
            if int(cls) == 0:  
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                bboxes.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        tracks_info = tracker.update_tracks(bboxes, frame=frame)

        for track in tracks_info:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            crop = frame[int(t):int(b), int(l):int(r)]
            if crop.size == 0:
                continue
            embedding = extract_features(crop)
            tracks.append([frame_id, track_id, l, t, r, b] + embedding.tolist())

        frame_id += 1

    cap.release()
    df = pd.DataFrame(tracks)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved tracking results to: {output_csv}")

def match_players(broadcast_csv, tacticam_csv, output_csv):
    df_b = pd.read_csv(broadcast_csv, header=None)
    df_t = pd.read_csv(tacticam_csv, header=None)

    def get_embeddings(df):
        grouped = df.groupby(1)
        embeddings = {}
        for pid, group in grouped:
            embs = group.iloc[:, 6:].values
            embeddings[pid] = np.mean(embs, axis=0)
        return embeddings

    emb_b = get_embeddings(df_b)
    emb_t = get_embeddings(df_t)

    matches = []
    for tid, t_emb in emb_t.items():
        best_match = None
        best_score = float("inf")
        for bid, b_emb in emb_b.items():
            dist = cosine(t_emb, b_emb)
            if dist < best_score:
                best_score = dist
                best_match = bid
        matches.append([tid, best_match, round(1 - best_score, 3)])

    df_matches = pd.DataFrame(matches, columns=["tacticam_player_id", "broadcast_player_id", "similarity"])
    df_matches.to_csv(output_csv, index=False)
    print(f"[INFO] Saved player match results to: {output_csv}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("[STEP 1] Processing broadcast video...")
    detect_and_track("broadcast.mp4", "outputs/broadcast_tracks.csv")

    print("[STEP 2] Processing tacticam video...")
    detect_and_track("tacticam.mp4", "outputs/tacticam_tracks.csv")

    print("[STEP 3] Matching players across videos...")
    match_players("outputs/broadcast_tracks.csv", "outputs/tacticam_tracks.csv", "outputs/matched_players.csv")
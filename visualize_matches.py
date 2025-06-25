import cv2
import pandas as pd
import numpy as np
import os


matches = pd.read_csv("outputs/matched_players.csv")


broadcast_df = pd.read_csv("outputs/broadcast_tracks.csv", header=None)
tacticam_df = pd.read_csv("outputs/tacticam_tracks.csv", header=None)


broadcast_cap = cv2.VideoCapture("broadcast.mp4")
tacticam_cap = cv2.VideoCapture("tacticam.mp4")

def get_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    return frame if ret else None


def get_color(id):
    np.random.seed(int(id))
    return tuple(np.random.randint(100, 255, size=3).tolist())


for idx, row in matches.iterrows():
    t_id = row['tacticam_player_id']
    b_id = row['broadcast_player_id']
    sim = row['similarity']

    print(f"Displaying match: Tacticam ID {t_id} ↔ Broadcast ID {b_id} (Similarity: {sim})")

    
    t_frames = tacticam_df[tacticam_df[1] == t_id]
    b_frames = broadcast_df[broadcast_df[1] == b_id]

    if t_frames.empty or b_frames.empty:
        print(f"Skipping ID {t_id} or {b_id} due to missing frames.")
        continue

    
    t_row = t_frames.iloc[0]
    b_row = b_frames.iloc[0]

    
    t_frame = get_frame(tacticam_cap, int(t_row[0]))
    b_frame = get_frame(broadcast_cap, int(b_row[0]))

    if t_frame is None or b_frame is None:
        continue

    
    t_box = tuple(map(int, [t_row[2], t_row[3], t_row[4], t_row[5]]))
    b_box = tuple(map(int, [b_row[2], b_row[3], b_row[4], b_row[5]]))

    t_color = get_color(t_id)
    b_color = get_color(b_id)

    cv2.rectangle(t_frame, (t_box[0], t_box[1]), (t_box[2], t_box[3]), t_color, 2)
    cv2.putText(t_frame, f"T_ID {t_id}", (t_box[0], t_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 2)

    cv2.rectangle(b_frame, (b_box[0], b_box[1]), (b_box[2], b_box[3]), b_color, 2)
    cv2.putText(b_frame, f"B_ID {b_id}", (b_box[0], b_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_color, 2)

    
    t_resized = cv2.resize(t_frame, (640, 360))
    b_resized = cv2.resize(b_frame, (640, 360))

    combined = np.hstack((b_resized, t_resized))
    cv2.imshow(f'Match: Broadcast {b_id} ↔ Tacticam {t_id}', combined)

    key = cv2.waitKey(0)
    if key == 27:  
        break

cv2.destroyAllWindows()
broadcast_cap.release()
tacticam_cap.release()

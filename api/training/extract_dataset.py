"""
Extract MediaPipe keypoints from downloaded WLASL videos.

Usage:
  python extract_dataset.py asl100
  python extract_dataset.py asl300
"""

import sys, json, os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()

SPLIT      = sys.argv[1] if len(sys.argv) > 1 else 'asl100'
assert SPLIT in ('asl100', 'asl300', 'asl1000', 'asl2000'), f"Unknown split: {SPLIT}"
print(f"Extracting keypoints for: {SPLIT}")

WLASL_DIR     = os.getenv("WLASL_DIR",     "../WLASL")
MODEL_PATH_POSE = os.getenv("MODEL_PATH_POSE", "model/pose_landmarker_lite.task")
MODEL_PATH_HAND = os.getenv("MODEL_PATH_HAND", "model/hand_landmarker.task")

SPLIT_FILE    = os.path.join(WLASL_DIR, "data", "splits", f"{SPLIT}.json")
VIDEOS_DIR    = os.path.join(WLASL_DIR, "videos")
HF_VIDEOS_DIR = os.path.join(WLASL_DIR, "hf_videos")
OUTPUT_DIR    = os.path.join(WLASL_DIR, "keypoints")
NUM_FRAMES    = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SPLIT_FILE) as f:
    data = json.load(f)

word_labels  = sorted([d['gloss'] for d in data])
label_to_idx = {w: i for i, w in enumerate(word_labels)}

video_index = {}
for entry in data:
    for inst in entry['instances']:
        video_index[inst['video_id']] = {
            'gloss':       entry['gloss'],
            'label_idx':   label_to_idx[entry['gloss']],
            'split':       inst['split'],
            'frame_start': inst['frame_start'],
            'frame_end':   inst['frame_end'],
        }

def find_video(video_id):
    p = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    for root, dirs, files in os.walk(HF_VIDEOS_DIR):
        if f"{video_id}.mp4" in files:
            p = os.path.join(root, f"{video_id}.mp4")
            if os.path.getsize(p) > 1000:
                return p
    return None

print("Building video file index...")
available = {vid: find_video(vid) for vid in video_index if find_video(vid)}
print(f"Videos available: {len(available)}/{len(video_index)}")

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_POSE),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.3,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
)

def normalize(x, y): return 2*x-1, 2*y-1

def extract_body(pose_lms):
    kp = np.zeros((13, 2))
    if not pose_lms: return kp
    lm = pose_lms[0]
    def n(i):
        nx, ny = normalize(lm[i].x, lm[i].y)
        return np.array([nx, ny])
    def avg(a,b): return (n(a)+n(b))/2
    kp[0]=n(0); kp[1]=avg(11,12); kp[2]=n(12); kp[3]=n(14); kp[4]=n(16)
    kp[5]=n(11); kp[6]=n(13); kp[7]=n(15); kp[8]=avg(23,24)
    kp[9]=n(5); kp[10]=n(2); kp[11]=n(8); kp[12]=n(7)
    return kp

def extract_hands(hand_lms, handedness_list):
    left = np.zeros((21,2)); right = np.zeros((21,2))
    if not hand_lms: return left, right
    for hlms, handed in zip(hand_lms, handedness_list):
        kp = np.array([[*(normalize(lm.x, lm.y))] for lm in hlms])
        if handed[0].category_name == 'Left': left = kp
        else: right = kp
    return left, right

def process_video(video_path, frame_start, frame_end):
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = max(0, frame_start - 1)
    fe = min(total_frames - 1, frame_end - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fs)
    frames = []
    while cap.isOpened():
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > fe: break
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    if not frames: return None

    total   = len(frames)
    indices = (np.linspace(0, total-1, NUM_FRAMES, dtype=int)
               if total >= NUM_FRAMES
               else list(range(total)) + [total-1]*(NUM_FRAMES-total))
    sampled = [frames[i] for i in indices]

    all_kp = []
    with vision.PoseLandmarker.create_from_options(pose_options) as pd, \
         vision.HandLandmarker.create_from_options(hand_options) as hd:
        for frame in sampled:
            h, w   = frame.shape[:2]
            size   = min(h, w)
            frame  = frame[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
            frame  = cv2.resize(frame, (256,256))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pr = pd.detect(mp_img); hr = hd.detect(mp_img)
            body   = extract_body(pr.pose_landmarks)
            lh, rh = extract_hands(hr.hand_landmarks, hr.handedness)
            all_kp.append(np.concatenate([body, lh, rh], axis=0))

    arr    = np.array(all_kp)
    tensor = np.zeros((55, NUM_FRAMES * 2))
    for t in range(NUM_FRAMES):
        tensor[:, t*2]   = arr[t, :, 0]
        tensor[:, t*2+1] = arr[t, :, 1]
    return tensor

metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
metadata = json.load(open(metadata_path)) if os.path.exists(metadata_path) else {}

success = failed = skipped = 0
for i, (video_id, video_path) in enumerate(available.items()):
    out_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")
    if os.path.exists(out_path):
        metadata[video_id] = video_index[video_id]
        success += 1; skipped += 1
        continue
    info = video_index[video_id]
    print(f"[{i+1}/{len(available)}] {video_id} ({info['gloss']})...")
    try:
        tensor = process_video(video_path, info['frame_start'], info['frame_end'])
        if tensor is None: raise Exception("No frames extracted")
        np.save(out_path, tensor)
        metadata[video_id] = info
        success += 1
        print(f"  ✓ {tensor.shape}")
    except Exception as e:
        print(f"  ✗ {e}"); failed += 1

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone! {success} processed ({skipped} skipped), {failed} failed")
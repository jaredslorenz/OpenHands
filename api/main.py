import os
os.environ["GLOG_minloglevel"] = "3"  # suppress MediaPipe logs

import io
import json
import tempfile
import base64

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SiglipForImageClassification
from dotenv import load_dotenv

from extract_keypoints import extract_keypoints_from_video

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Letter model ──────────────────────────────────────────────────────────────
letter_model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
letter_model      = SiglipForImageClassification.from_pretrained(letter_model_name)
letter_processor  = AutoImageProcessor.from_pretrained(letter_model_name)
letter_model.eval()

LETTER_LABELS = {
    0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",
    8:"I",9:"J",10:"K",11:"L",12:"M",13:"N",14:"O",
    15:"P",16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",
    22:"W",23:"X",24:"Y",25:"Z"
}

# ── Word classifier — v7 architecture ────────────────────────────────────────
NUM_JOINTS  = 55
IN_CHANNELS = 4
NUM_FRAMES  = 50
TTA_RUNS    = 8

class ASLClassifier(nn.Module):
    def __init__(self, input_size=NUM_JOINTS*IN_CHANNELS,
                 hidden_size=384, num_layers=2,
                 num_classes=100, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attn(out), dim=1)
        out     = (out * attn_w).sum(dim=1)
        return self.classifier(out)


# ── Config — all paths via env vars, relative defaults for deployment ─────────
MODEL_PATH_POSE = os.getenv("MODEL_PATH_POSE", "model/pose_landmarker_lite.task")
MODEL_PATH_HAND = os.getenv("MODEL_PATH_HAND", "model/hand_landmarker.task")
WORD_MODEL_PATH = os.getenv("WORD_MODEL_PATH", "lstm_model_v7.pth")
SPLIT_FILE      = os.getenv("SPLIT_FILE",      "../WLASL/data/splits/asl100.json")

with open(SPLIT_FILE) as f:
    word_labels = sorted([d['gloss'] for d in json.load(f)])

word_model = ASLClassifier(num_classes=len(word_labels))
ckpt = torch.load(WORD_MODEL_PATH, map_location='cpu')
if 'model_state_dict' in ckpt:
    word_model.load_state_dict(ckpt['model_state_dict'])
else:
    word_model.load_state_dict(ckpt)
word_model.eval()
print(f"Loaded: {WORD_MODEL_PATH}  |  val_acc: 69.2%")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_POSE),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.3,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)


# ── Feature helpers ───────────────────────────────────────────────────────────
def compute_velocity(frames: torch.Tensor) -> torch.Tensor:
    vel = frames[1:] - frames[:-1]
    vel = torch.cat([vel, vel[-1:].clone()], dim=0)
    return torch.cat([frames, vel], dim=-1)


def prepare_input(tensor: torch.Tensor) -> torch.Tensor:
    t      = tensor.squeeze(0)
    frames = torch.zeros(NUM_FRAMES, NUM_JOINTS, 2)
    for i in range(NUM_FRAMES):
        frames[i, :, 0] = t[:, i*2]
        frames[i, :, 1] = t[:, i*2+1]
    frames = compute_velocity(frames)
    return frames.reshape(1, NUM_FRAMES, NUM_JOINTS * IN_CHANNELS)


def tta_augment(frames: torch.Tensor) -> torch.Tensor:
    f  = frames.clone().reshape(NUM_FRAMES, NUM_JOINTS, IN_CHANNELS)
    xy = f[:, :, :2]
    xy = xy * torch.empty(1).uniform_(0.95, 1.05).item()
    theta   = np.radians(np.random.uniform(-8, 8))
    rot     = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]], dtype=xy.dtype)
    xy      = xy @ rot.T
    xy      = xy + torch.randn_like(xy) * 0.008
    vel_new = torch.cat([xy[1:] - xy[:-1], (xy[-1:] - xy[-2:-1]).clone()], dim=0)
    f_aug   = torch.cat([xy, vel_new], dim=-1)
    return f_aug.reshape(1, NUM_FRAMES, NUM_JOINTS * IN_CHANNELS)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict/letter")
async def predict_letter(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w, h  = image.size
    image = image.crop((int(w*0.1), int(h*0.25), int(w*0.9), int(h*0.65)))
    inputs = letter_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        probs = torch.nn.functional.softmax(
            letter_model(**inputs).logits, dim=1).squeeze()
    pred_idx = int(torch.argmax(probs))
    return {"letter": LETTER_LABELS[pred_idx], "confidence": float(probs[pred_idx])}


@app.post("/predict/word")
async def predict_word(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        tensor = extract_keypoints_from_video(tmp_path)
        if tensor is None:
            return {"error": "Could not extract keypoints from video"}

        base_input = prepare_input(tensor)

        with torch.no_grad():
            all_probs = torch.softmax(word_model(base_input), dim=1)
            for _ in range(TTA_RUNS):
                all_probs += torch.softmax(word_model(tta_augment(base_input)), dim=1)

        probs      = (all_probs / (TTA_RUNS + 1)).squeeze()
        pred_idx   = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])

        top5 = torch.topk(probs, 5)
        print("\n── Word Prediction (TTA) ────────────────")
        for rank, (idx, prob) in enumerate(zip(top5.indices, top5.values)):
            marker = " ← predicted" if rank == 0 else ""
            print(f"  {rank+1}. {word_labels[int(idx)]:<20} {float(prob)*100:.1f}%{marker}")
        print(f"  (averaged over {TTA_RUNS + 1} passes)")
        print("─────────────────────────────────────────\n")

        return {
            "word":       word_labels[pred_idx],
            "confidence": confidence,
            "top5": [
                {"word": word_labels[int(idx)], "confidence": float(prob)}
                for idx, prob in zip(top5.indices, top5.values)
            ]
        }
    finally:
        os.unlink(tmp_path)


@app.post("/debug/keypoints")
async def debug_keypoints(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "Could not read video frame"}

        img_h, img_w   = frame.shape[:2]
        rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image       = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected_body  = False
        detected_hands = 0

        with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, \
             vision.HandLandmarker.create_from_options(hand_options) as hand_detector:

            pose_result = pose_detector.detect(mp_image)
            hand_result = hand_detector.detect(mp_image)

            if pose_result.pose_landmarks:
                detected_body = True
                UPPER_BODY    = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
                for idx in UPPER_BODY:
                    lm = pose_result.pose_landmarks[0][idx]
                    cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 8, (255,91,99), -1)
                lms = pose_result.pose_landmarks[0]
                for a,b in [(0,11),(0,12),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24)]:
                    cv2.line(frame,
                             (int(lms[a].x*img_w), int(lms[a].y*img_h)),
                             (int(lms[b].x*img_w), int(lms[b].y*img_h)),
                             (255,91,99), 2)

            if hand_result.hand_landmarks:
                detected_hands = len(hand_result.hand_landmarks)
                for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    color = (88,209,48) if handedness[0].category_name == 'Left' else (58,69,255)
                    for lm in hand_lms:
                        cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 5, color, -1)

        cv2.putText(frame,
                    f"Body: {'YES' if detected_body else 'NO'}  |  Hands: {detected_hands}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        _, buffer  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return {
            "image":          base64.b64encode(buffer).decode('utf-8'),
            "detected_body":  detected_body,
            "detected_hands": detected_hands
        }
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok", "model": "lstm_v7", "val_acc": "69.2%", "tta_runs": TTA_RUNS}
"""
config.py — centralised configuration and model loading.
All environment variables, model initialisation, and MediaPipe
options live here so every other module imports from one place.
"""

import os
os.environ["GLOG_minloglevel"] = "3"  # suppress MediaPipe GL logs

import json
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

from models import ASLClassifier

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH_POSE = os.getenv("MODEL_PATH_POSE", "model/pose_landmarker_lite.task")
MODEL_PATH_HAND = os.getenv("MODEL_PATH_HAND", "model/hand_landmarker.task")
WORD_MODEL_PATH = os.getenv("WORD_MODEL_PATH_100", "lstm_model_v7.pth")
LABELS_FILE     = os.getenv("LABELS_FILE_100",     "data/labels/word_labels_100.json")

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_JOINTS  = 55
IN_CHANNELS = 4   # x, y, dx, dy
NUM_FRAMES  = 50
TTA_RUNS    = 8

LETTER_LABELS = {
    0:"A",  1:"B",  2:"C",  3:"D",  4:"E",  5:"F",  6:"G",  7:"H",
    8:"I",  9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P",
   16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X",
   24:"Y", 25:"Z"
}

# ── Word labels ───────────────────────────────────────────────────────────────
with open(LABELS_FILE) as f:
    word_labels: list[str] = json.load(f)

# ── Word model ────────────────────────────────────────────────────────────────
word_model = ASLClassifier(num_classes=len(word_labels))
_ckpt = torch.load(WORD_MODEL_PATH, map_location="cpu")
word_model.load_state_dict(
    _ckpt["model_state_dict"] if "model_state_dict" in _ckpt else _ckpt
)
word_model.eval()
print(f"[config] Loaded word model: {WORD_MODEL_PATH}  |  val_acc: 69.2%")

# ── Letter model (lazy-loaded on first request) ───────────────────────────────
# The HuggingFace model is large (~900 MB). Loading it lazily avoids consuming
# memory on free-tier deployments when /predict/letter is not being used.
_letter_model     = None
_letter_processor = None

def get_letter_model():
    """Return the letter model, loading it on first call."""
    global _letter_model, _letter_processor
    if _letter_model is None:
        from transformers import AutoImageProcessor, SiglipForImageClassification
        _letter_model     = SiglipForImageClassification.from_pretrained(
            "prithivMLmods/Alphabet-Sign-Language-Detection"
        )
        _letter_processor = AutoImageProcessor.from_pretrained(
            "prithivMLmods/Alphabet-Sign-Language-Detection"
        )
        _letter_model.eval()
        print("[config] Letter model loaded on first request")
    return _letter_model, _letter_processor

# ── MediaPipe options ─────────────────────────────────────────────────────────
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
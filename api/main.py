"""
main.py — FastAPI application entry point.
Route definitions and HTTP concerns only.
Business logic lives in inference.py and keypoints.py.
"""

import os
import base64
import tempfile

import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from inference import predict_letter_from_image, predict_word_from_tensor
from extract_keypoints import extract_keypoints_from_video

app = FastAPI(
    title="OpenHands ASL API",
    description="Real-time ASL letter and word recognition via MediaPipe + BiLSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mobile clients don't send Origin headers — CORS not applicable
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── File validation ───────────────────────────────────────────────────────────
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 50 MB
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_file(contents: bytes, max_size: int, allowed_types: set, content_type: str) -> str | None:
    """Return an error string if the file is invalid, else None."""
    if len(contents) > max_size:
        return f"File too large. Maximum size is {max_size // (1024*1024)}MB"
    if content_type not in allowed_types:
        return f"Unsupported file type: {content_type}"
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict/letter")
async def predict_letter(file: UploadFile = File(...)):
    contents = await file.read()

    error = validate_file(contents, MAX_IMAGE_SIZE, ALLOWED_IMAGE_TYPES, file.content_type)
    if error:
        return {"error": error}

    return predict_letter_from_image(contents)


@app.post("/predict/word")
async def predict_word(file: UploadFile = File(...)):
    contents = await file.read()

    error = validate_file(contents, MAX_VIDEO_SIZE, ALLOWED_VIDEO_TYPES, file.content_type)
    if error:
        return {"error": error}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        tensor = extract_keypoints_from_video(tmp_path)
        if tensor is None:
            return {"error": "Could not extract keypoints — ensure hands are visible"}
        return predict_word_from_tensor(tensor)
    finally:
        os.unlink(tmp_path)


@app.post("/debug/keypoints")
async def debug_keypoints(file: UploadFile = File(...)):
    contents = await file.read()

    error = validate_file(contents, MAX_VIDEO_SIZE, ALLOWED_VIDEO_TYPES, file.content_type)
    if error:
        return {"error": error}

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

        with mp.tasks.vision.PoseLandmarker.create_from_options(config.pose_options) as pose_detector, \
             mp.tasks.vision.HandLandmarker.create_from_options(config.hand_options) as hand_detector:

            pose_result = pose_detector.detect(mp_image)
            hand_result = hand_detector.detect(mp_image)

            if pose_result.pose_landmarks:
                detected_body = True
                UPPER_BODY    = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
                for idx in UPPER_BODY:
                    lm = pose_result.pose_landmarks[0][idx]
                    cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 8, (255, 91, 99), -1)
                lms = pose_result.pose_landmarks[0]
                for a, b in [(0,11),(0,12),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24)]:
                    cv2.line(frame,
                             (int(lms[a].x*img_w), int(lms[a].y*img_h)),
                             (int(lms[b].x*img_w), int(lms[b].y*img_h)),
                             (255, 91, 99), 2)

            if hand_result.hand_landmarks:
                detected_hands = len(hand_result.hand_landmarks)
                for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    color = (88, 209, 48) if handedness[0].category_name == "Left" else (58, 69, 255)
                    for lm in hand_lms:
                        cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 5, color, -1)

        cv2.putText(
            frame,
            f"Body: {'YES' if detected_body else 'NO'}  |  Hands: {detected_hands}",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
        )
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        return {
            "image":          base64.b64encode(buffer).decode("utf-8"),
            "detected_body":  detected_body,
            "detected_hands": detected_hands,
        }
    finally:
        os.unlink(tmp_path)


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "model":    "lstm_v7",
        "val_acc":  "69.2%",
        "tta_runs": config.TTA_RUNS,
    }
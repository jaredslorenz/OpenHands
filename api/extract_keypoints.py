import os
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH_POSE  = os.getenv("MODEL_PATH_POSE", "model/pose_landmarker_lite.task")
MODEL_PATH_HAND  = os.getenv("MODEL_PATH_HAND", "model/hand_landmarker.task")
NUM_SAMPLES      = 50
HAND_CONF_THRESH = 0.5
SMOOTH_WINDOW    = 3


def normalize(x, y):
    return 2 * x - 1, 2 * y - 1


def extract_body_keypoints(pose_landmarks):
    kp = np.zeros((13, 2))
    if not pose_landmarks:
        return kp
    lm = pose_landmarks[0]
    def n(idx):
        nx, ny = normalize(lm[idx].x, lm[idx].y)
        return np.array([nx, ny])
    def avg(a, b): return (n(a) + n(b)) / 2
    kp[0]=n(0); kp[1]=avg(11,12); kp[2]=n(12); kp[3]=n(14); kp[4]=n(16)
    kp[5]=n(11); kp[6]=n(13); kp[7]=n(15); kp[8]=avg(23,24)
    kp[9]=n(5); kp[10]=n(2); kp[11]=n(8); kp[12]=n(7)
    return kp


def extract_hand_keypoints(hand_landmarks, handedness_list):
    left = np.zeros((21, 2))
    right = np.zeros((21, 2))
    if not hand_landmarks:
        return left, right
    for hand_lms, handedness in zip(hand_landmarks, handedness_list):
        if handedness[0].score < HAND_CONF_THRESH:
            continue
        kp = np.zeros((21, 2))
        for i, lm in enumerate(hand_lms):
            kp[i] = normalize(lm.x, lm.y)
        if handedness[0].category_name == 'Left':
            left = kp
        else:
            right = kp
    return left, right


def has_hands(keypoints):
    return (np.abs(keypoints[13:34]).sum() > 0) or (np.abs(keypoints[34:55]).sum() > 0)


def smooth_keypoints(keypoints_list, window=SMOOTH_WINDOW):
    if len(keypoints_list) < window:
        return keypoints_list
    arr      = np.array(keypoints_list)
    smoothed = arr.copy()
    half     = window // 2
    for t in range(len(arr)):
        t_start = max(0, t - half)
        t_end   = min(len(arr), t + half + 1)
        window_frames = arr[t_start:t_end]
        detected_mask = np.abs(window_frames).sum(axis=-1) > 0
        for j in range(55):
            detected = window_frames[detected_mask[:, j], j, :]
            if len(detected) > 0:
                smoothed[t, j] = detected.mean(axis=0)
    return list(smoothed)


def trim_dead_frames(keypoints_list):
    if len(keypoints_list) <= 10:
        return keypoints_list
    start = next((i for i, kp in enumerate(keypoints_list) if has_hands(kp)), 0)
    end   = next((i for i in range(len(keypoints_list)-1, -1, -1) if has_hands(keypoints_list[i])), len(keypoints_list)-1)
    if end - start < 10:
        return keypoints_list
    return keypoints_list[start:end+1]


def extract_keypoints_from_video(video_path, num_samples=NUM_SAMPLES):
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

    cap            = cv2.VideoCapture(video_path)
    all_keypoints  = []

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_detector, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w   = frame.shape[:2]
            size   = min(h, w)
            frame  = frame[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
            frame  = cv2.resize(frame, (256, 256))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pr = pose_detector.detect(mp_img)
            hr = hand_detector.detect(mp_img)

            body     = extract_body_keypoints(pr.pose_landmarks)
            lh, rh   = extract_hand_keypoints(hr.hand_landmarks, hr.handedness)
            all_keypoints.append(np.concatenate([body, lh, rh], axis=0))

    cap.release()
    if not all_keypoints:
        return None

    all_keypoints = trim_dead_frames(all_keypoints)
    all_keypoints = smooth_keypoints(all_keypoints)

    total   = len(all_keypoints)
    indices = (np.linspace(0, total-1, num_samples, dtype=int)
               if total >= num_samples
               else list(range(total)) + [total-1] * (num_samples - total))
    sampled = np.array([all_keypoints[i] for i in indices])

    tensor = np.zeros((55, num_samples * 2))
    for t in range(num_samples):
        tensor[:, t*2]   = sampled[t, :, 0]
        tensor[:, t*2+1] = sampled[t, :, 1]

    return torch.FloatTensor(tensor).unsqueeze(0)


if __name__ == '__main__':
    print("Keypoint extraction module ready")
    print(f"Pose model : {MODEL_PATH_POSE}")
    print(f"Hand model : {MODEL_PATH_HAND}")
    print(f"Hand conf  : {HAND_CONF_THRESH}")
    print(f"Smooth     : {SMOOTH_WINDOW} frames")
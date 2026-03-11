# OpenHands 🤝

**Real-time American Sign Language (ASL) to text translation, built from scratch.**

A full-stack ML mobile application that translates ASL gestures into text in real time. Built with a custom-trained BiLSTM model on the WLASL dataset, a FastAPI backend, and a React Native/Expo mobile app.

---

## Demo

> Letter mode detects individual ASL letters in real time via photo snapshots.
> Word mode records a 2-second video clip, extracts skeleton keypoints, and predicts the signed word.

---

## Architecture Overview

```
iPhone Camera
     │
     ▼
React Native App (Expo)
     │
     │  HTTP (photo / video)
     ▼
FastAPI Backend (Mac)
     │
     ├── /predict/letter → HuggingFace SiglipForImageClassification → A-Z
     │
     └── /predict/word  → MediaPipe Keypoint Extraction
                               │
                               ▼
                         BiLSTM Classifier
                               │
                               ▼
                         Top prediction + confidence
```

---

## ML Pipeline

### Dataset

- **WLASL (World Level American Sign Language)** — the largest publicly available ASL video dataset
- Original dataset: 21,095 videos across 2,000 words
- Downloaded via [Voxel51/WLASL on HuggingFace](https://huggingface.co/datasets/Voxel51/WLASL) — 11,880 accessible videos
- Final extracted keypoints: **12,144 videos** across all splits
- Training subset: **asl100** — 100 words, ~1,200 keypoint files

### Keypoint Extraction

- **MediaPipe** — chosen over OpenPose for mobile compatibility and reliability
- 55 joints total: 13 upper body + 21 left hand + 21 right hand
- Each video sampled to exactly 50 frames
- Preprocessing: square center crop → resize to 256×256 → normalize to [-1, 1]
- Output format: `(55, 100)` array — 55 joints × 50 frames × 2 coords (interleaved)

### Model Architecture — BiLSTM with Attention

```
Input: (50, 220) — 50 frames × (55 joints × 4 features)
         │
         ▼
Bidirectional LSTM (hidden=384, layers=2, dropout=0.4)
         │
         ▼
Attention Pooling — learns which frames matter most
         │
         ▼
LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→100)
         │
         ▼
100-class softmax
```

### Feature Engineering

Each joint has 4 features per frame:

- `x, y` — raw normalized coordinates
- `dx, dy` — velocity (difference from previous frame)

Signs are fundamentally about **motion**, not just pose. Adding velocity as an explicit feature was one of the biggest single accuracy improvements.

### Augmentation Pipeline

After extensive experimentation, the following augmentations were used in the final model:

- **Gaussian jitter** — small noise added to coordinates
- **Temporal shift** — roll sequence by ±2 frames
- **Scale jitter** — random scale between 0.95–1.05
- **Rotation** — random ±10° rotation around Z-axis
- **Time warp** — stretch/compress random segments to simulate signing speed variation
- **Joint dropout** — zero entire joints (40% chance) to simulate occlusion
- **Cutout** — zero 3 random temporal windows of 5 frames each
- **Mixup** — blend two samples with mixed labels (50% of batches, α=0.2)

### Training

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
- Scheduler: OneCycleLR (per batch)
- Loss: CrossEntropyLoss (label_smoothing=0.1)
- Epochs: up to 120 with patience=15
- Hardware: Apple MPS (MacBook)

---

## Model Performance History

| Model  | Val Accuracy | Notes                                      |
| ------ | ------------ | ------------------------------------------ |
| v1     | 26.7%        | Baseline, raw coords, 571 videos           |
| v2     | 19.8%        | Too aggressive augmentation                |
| v3     | 56.5%        | 1,271 videos, mild augmentation            |
| v4     | 25.7%        | Body-centered normalization — hurt LSTM    |
| v5     | 58.4%        | Velocity features + attention + rotation   |
| v6     | ~58%         | Extreme augmentation                       |
| **v7** | **69.2%**    | Mixup + time warp + joint dropout + cutout |
| v8     | 69.2%        | Larger model (512 hidden) — tied v7        |

**Best model: `lstm_model_v7.pth` — 69.2% validation accuracy on 100 words**

---

## Design Decisions & What We Tried

### MediaPipe over OpenPose

OpenPose was initially considered since the original WLASL paper used it and more pretrained models exist. However:

- OpenPose requires CUDA and is painful to install on Mac
- OpenPose uses pixel coordinates in full image space; MediaPipe uses normalized 0-1 coordinates relative to a crop
- The coordinate system mismatch made it impossible to bridge the two with a learned mapping network
- MediaPipe runs natively on-device and is mobile-compatible

**Decision: Pure MediaPipe pipeline.**

### Why BiLSTM over ST-GCN

A Spatial-Temporal Graph Convolutional Network (ST-GCN) was implemented and tested extensively:

- Pure ST-GCN: ~14% val accuracy
- GCN-LSTM hybrid: ~20% val accuracy
- Both significantly underperformed the BiLSTM

The BiLSTM's simplicity is an advantage on small datasets — ST-GCN has too many parameters relative to ~1,200 training videos. GCNs shine with large datasets where the graph structure provides meaningful inductive bias.

### Why 100 Words

Training was attempted on asl300, asl1000, and asl2000 splits:

- asl300: 35.9% val (overfitting — not enough data per class)
- asl1000: 20.7% val (severe overfitting)
- asl100 consistently outperforms larger splits with available data

More classes require more data per class. With ~50% of WLASL videos being dead links, asl100 is the practical sweet spot.

### Feature Normalization

Body-centered normalization (subtracting shoulder midpoint, dividing by shoulder distance) was tested but **hurt** the BiLSTM. Reason: absolute position is meaningful in ASL — signs like "mother" involve touching specific face locations. Removing position information removes a key discriminating signal for the LSTM.

The TGCN benefited from normalization; the LSTM did not.

### Velocity Features

Adding `(dx, dy)` alongside `(x, y)` gave a consistent accuracy boost across all model versions. Signs are defined by motion — a static pose is ambiguous, but the trajectory is distinctive. This was the single most impactful feature engineering change.

### Mixup

Blending two training samples with mixed labels (Mixup) was the biggest single augmentation improvement. Combined with time warp, joint dropout, and cutout, it pushed val accuracy from 58% to 69.2%.

---

## Stack

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| Mobile App          | React Native, Expo SDK 54, Expo Router   |
| Camera              | react-native-vision-camera               |
| Backend             | FastAPI, Python 3.10                     |
| Keypoint Extraction | MediaPipe (pose + hand landmarkers)      |
| ML Framework        | PyTorch                                  |
| Letter Model        | HuggingFace SiglipForImageClassification |
| Word Model          | Custom BiLSTM                            |
| Environment         | Conda (openhands-api)                    |

---

## Project Structure

```
OpenHands/
├── api/
│   ├── main.py                  # FastAPI server
│   ├── extract_keypoints.py     # MediaPipe keypoint extraction (inference)
│   ├── lstm_model_v7.pth        # Best word model (69.2% val)
│   ├── word_labels_100.json     # 100-word label list
│   └── model/
│       ├── pose_landmarker_lite.task
│       └── hand_landmarker.task
├── mobile/
│   ├── app/
│   │   └── (tabs)/
│   │       ├── index.tsx        # Home screen
│   │       └── camera.tsx       # Camera + prediction UI
│   └── .env                     # EXPO_PUBLIC_API_URL
├── WLASL/
│   ├── data/splits/             # asl100/300/1000/2000.json
│   ├── keypoints/               # Extracted .npy files (gitignored)
│   └── hf_videos/               # Downloaded videos (gitignored)
├── extract_dataset.py           # Batch keypoint extraction
├── train.py                     # BiLSTM training (v7 settings)
└── train_tgcn.py                # ST-GCN experiments
```

---

## Setup

### Backend

```bash
conda activate openhands-api
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Mobile App

```bash
cd mobile
# Set your local IP in .env
echo "EXPO_PUBLIC_API_URL=http://YOUR_IP:8000" > .env
npx expo run:ios
```

Find your IP with:

```bash
ipconfig getifaddr en0
```

### Training

```bash
# Extract keypoints
python extract_dataset.py asl100

# Train
python train.py
```

---

## API Endpoints

| Endpoint           | Method | Input       | Output                           |
| ------------------ | ------ | ----------- | -------------------------------- |
| `/predict/letter`  | POST   | image (jpg) | `{letter, confidence}`           |
| `/predict/word`    | POST   | video (mp4) | `{word, confidence, top5}`       |
| `/debug/keypoints` | POST   | video (mp4) | `{image}` base64 annotated frame |

---

## License

WLASL dataset is licensed under the C-UDA license — **academic use only.**

---

## Acknowledgements

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Dongxu Li et al.
- [Voxel51/WLASL on HuggingFace](https://huggingface.co/datasets/Voxel51/WLASL)
- [MediaPipe](https://developers.google.com/mediapipe)

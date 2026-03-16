# OpenHands 🤝

**Real-time American Sign Language (ASL) to text translation, built from scratch.**

A full-stack ML mobile application that translates ASL gestures into text in real time. Built with a custom-trained BiLSTM model on the WLASL benchmark dataset, a modular FastAPI backend, and a React Native/Expo mobile app.

---

## Demo

> **Letter mode** — takes a photo snapshot every 500ms and predicts the signed letter in real time.
> **Word mode** — records a 2-second video clip, extracts MediaPipe skeleton keypoints, and predicts the signed word using BiLSTM + test-time augmentation.

---

## Architecture Overview

```
iPhone Camera
     │
     ▼
React Native App (Expo)
     │
     │  HTTP multipart (photo / video)
     ▼
FastAPI Backend (Render)
     │
     ├── /predict/letter → HuggingFace SiglipForImageClassification → A-Z
     │
     └── /predict/word  → MediaPipe Keypoint Extraction
                               │
                               ▼
                         BiLSTM v7 Classifier
                         (TTA — 8 augmented passes averaged)
                               │
                               ▼
                         word + confidence + top 5
```

---

## ML Pipeline

### Dataset

- **WLASL (World Level American Sign Language)** — the largest publicly available ASL video dataset
- Original dataset: 21,095 videos across 2,000 words
- Downloaded via [Voxel51/WLASL on HuggingFace](https://huggingface.co/datasets/Voxel51/WLASL) — 11,880 accessible videos
- Final extracted keypoints: **12,144 videos** across all splits
- Deployed on: **asl100** — 100 words, ~1,200 keypoint files (~12 videos/class)

### Keypoint Extraction

- **MediaPipe** pose + hand landmarkers — chosen over OpenPose for mobile compatibility
- 55 joints total: 13 upper body + 21 left hand + 21 right hand
- Each video sampled to exactly 50 frames
- Preprocessing: square center crop → resize to 256×256 → normalize to [-1, 1]
- Improvements over naive extraction: dead frame trimming, 3-frame moving average smoothing, hand confidence threshold (0.5)
- Output format: `(55, 100)` — 55 joints × 50 frames × 2 coords (interleaved)

### Model Architecture — BiLSTM with Attention (v7)

```
Input: (50, 220) — 50 frames × (55 joints × 4 features: x, y, dx, dy)
         │
         ▼
Bidirectional LSTM (hidden=384, layers=2, dropout=0.4)
         │
         ▼
Attention Pooling — learned per-frame weighting
         │
         ▼
LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→100)
         │
         ▼
100-class softmax
```

### Feature Engineering

Each joint has 4 features per frame: `x, y` (raw normalized coordinates) and `dx, dy` (velocity — difference from previous frame). Adding velocity was the single most impactful feature change — signs are defined by motion, not just pose. A static hand position is ambiguous; the trajectory is distinctive.

### Augmentation Pipeline (v7)

After extensive experimentation across 8 BiLSTM versions, the following stack gave the best results:

- **Gaussian jitter** — σ=0.01 coordinate noise
- **Temporal shift** — roll sequence by ±2 frames
- **Scale jitter** — random scale 0.95–1.05
- **Rotation** — random ±10° around Z-axis
- **Time warp** — stretch/compress segments to simulate signing speed variation (σ=0.15, 50% prob)
- **Joint dropout** — zero entire joints to simulate occlusion (40% prob, p=0.1)
- **Cutout** — zero 2 random temporal windows of 4 frames (40% prob)
- **Random noise dropout** — sparse zeroing (30% prob)
- **Mixup** — blend two samples with mixed labels (50% of batches, α=0.2)

### Inference — Test-Time Augmentation

At inference, the model runs 1 clean pass + 8 augmented passes (scale jitter, rotation, noise) and averages the softmax probabilities. This consistently improves real-world accuracy beyond what the validation numbers suggest.

### Training

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
- Scheduler: OneCycleLR (per batch)
- Loss: CrossEntropyLoss (label_smoothing=0.1)
- Epochs: up to 120 with early stopping (patience=15)
- Hardware: Apple MPS (MacBook)

---

## Model Experiments

Extensive architecture and augmentation research was conducted across four model families. Every experiment was run to completion with identical data splits for fair comparison.

### BiLSTM — Full Training History

| Model  | Val Acc   | Notes                                                                         |
| ------ | --------- | ----------------------------------------------------------------------------- |
| v1     | 26.7%     | Baseline, raw coords, 571 videos                                              |
| v3     | 56.5%     | 1,271 videos, mild augmentation, BiLSTM 256 hidden                            |
| v4     | 25.7%     | Body-centered normalization — HURT. Removes absolute position signs depend on |
| v5     | 58.4%     | Velocity features + attention pooling + rotation                              |
| v6     | ~58%      | Extreme augmentation — no gain                                                |
| **v7** | **69.2%** | **Mixup + time warp + joint dropout + cutout — BEST**                         |
| v8     | 69.2%     | 512 hidden, 3 layers — tied v7, not better. v7 size is optimal                |

### Multi-Vocabulary Scaling

| Split   | Classes | ~Videos | Val Acc   | Notes                                           |
| ------- | ------- | ------- | --------- | ----------------------------------------------- |
| asl100  | 100     | ~1,200  | **69.2%** | ~12 videos/class                                |
| asl300  | 300     | ~3,000  | 54.3%     | +18.4pts over previous best (35.9%) with v7 aug |
| asl2000 | 2,000   | ~12,000 | 27.3%     | ~6 videos/class — data bottleneck               |

Accuracy degrades as vocabulary grows because WLASL provides fewer videos per class at larger splits. The model never plateaued on asl2000 — it ran all 120 epochs — meaning the ceiling is data, not architecture.

### Architecture Comparison

Three architectures were tested with identical augmentation and data splits:

| Architecture             | Val Acc   | Key Finding                                             |
| ------------------------ | --------- | ------------------------------------------------------- |
| BiLSTM v7                | **69.2%** | Simplicity wins on small datasets                       |
| ST-GCN v2 (from scratch) | 16.4%     | Underfits — needs ~50x more data to learn graph weights |
| TGCN Pretrained v1       | 57.5%     | Pretrained adj. matrices gave +41pts over scratch       |
| TGCN Pretrained v2       | **60.7%** | Added BiLSTM temporal modeling + velocity features      |

### TGCN Transfer Learning

The original WLASL authors released a pretrained TGCN checkpoint (`ckpt.pth`) trained with OpenPose features. Rather than loading all weights (coordinate systems are incompatible), only the 41 learned **55×55 adjacency matrices** were extracted — these encode which joints co-vary for ASL signs, structural knowledge that transfers regardless of coordinate system.

```
ST-GCN from scratch:          16.4%
+ Pretrained adj. matrices:   57.5%   (+41.1 pts)
+ BiLSTM temporal modeling:   60.7%   (+3.2 pts)
```

Two-stage finetuning (freeze adj. matrices → unfreeze) was tested and **hurt** (47.7%) — the adjacency matrices and linear weights must co-adapt from the start.

### Key Lessons Learned

- **Augmentation strategy**: Mixup + time warp + joint dropout + cutout was the breakthrough — each technique targets a different failure mode and compounds
- **Body-centered normalization**: Hurt BiLSTM (56.5% → 25.7%) because absolute hand position is meaningful in ASL. Signs like "mother" involve touching specific face locations
- **Weighted sampling**: Hurt on asl100 — dataset is balanced enough. Oversampling rare classes caused overfitting
- **Horizontal flip**: Hurt — ASL signs are not mirror-symmetric
- **Model scaling**: v8 (512 hidden, 3 layers) tied v7 — data is the bottleneck, not capacity
- **Over-regularization**: Stacking dropout 0.4 + LSTM 192 + stages 8 + WD 2e-2 simultaneously caused underfitting — coordinated attacks must be measured
- **Two-stage finetuning**: att and linear weights must co-adapt together from the start
- **ST-GCN augmentation**: v7 augmentation on ST-GCN made it worse (16.4%) — you cannot augment around fundamental data starvation

---

## API

### Endpoints

| Endpoint           | Method | Input              | Output                               |
| ------------------ | ------ | ------------------ | ------------------------------------ |
| `/predict/letter`  | POST   | image (jpg, ≤10MB) | `{letter, confidence}`               |
| `/predict/word`    | POST   | video (mp4, ≤50MB) | `{word, confidence, top5}`           |
| `/debug/keypoints` | POST   | video (mp4)        | `{image}` base64 annotated frame     |
| `/health`          | GET    | —                  | `{status, model, val_acc, tta_runs}` |

### Backend Structure

The API is organized into focused modules — each file has one responsibility:

```
api/
├── main.py            # FastAPI app, routes, file validation
├── config.py          # Env vars, model loading, MediaPipe setup
├── models.py          # ASLClassifier architecture (importable by training scripts)
├── inference.py       # Prediction logic — letter and word
├── keypoints.py       # Feature helpers — velocity, TTA augmentation
└── extract_keypoints.py  # MediaPipe video → keypoint tensor
```

---

## Project Structure

```
OpenHands/
├── api/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── inference.py
│   ├── keypoints.py
│   ├── extract_keypoints.py
│   ├── lstm_model_v7.pth        # Deployed word model (69.2% val)
│   ├── requirements.txt
│   ├── data/
│   │   └── labels/
│   │       └── word_labels_100.json
│   └── model/
│       ├── pose_landmarker_lite.task
│       └── hand_landmarker.task
├── api/training/
│   ├── train.py                 # BiLSTM v7 — asl100
│   ├── train_300.py             # BiLSTM v7 — asl300
│   ├── train_2000.py            # BiLSTM v7 — asl2000
│   ├── train_tgcn_v2.py         # ST-GCN + full v7 augmentation
│   ├── train_hybrid.py          # Hybrid GCN+LSTM
│   ├── train_tgcn_pretrained.py # TGCN + pretrained adj. matrices (v1)
│   ├── train_tgcn_pretrained_v2.py # TGCN + BiLSTM temporal (v2, best TGCN)
│   └── extract_att_matrices.py  # Extract adj. matrices from WLASL checkpoint
├── mobile/
│   ├── app/(tabs)/
│   │   ├── index.tsx            # Home screen
│   │   └── camera.tsx           # Camera + prediction UI
│   └── .env                     # EXPO_PUBLIC_API_URL
└── WLASL/
    ├── data/splits/             # asl100/300/1000/2000.json
    ├── keypoints/               # Extracted .npy files (gitignored)
    └── hf_videos/               # Downloaded videos (gitignored)
```

---

## Stack

| Component           | Technology                                             |
| ------------------- | ------------------------------------------------------ |
| Mobile App          | React Native, Expo SDK 54, Expo Router                 |
| Camera              | react-native-vision-camera                             |
| Backend             | FastAPI, Python 3.10, deployed on Render               |
| Keypoint Extraction | MediaPipe (pose + hand landmarkers)                    |
| ML Framework        | PyTorch                                                |
| Letter Model        | HuggingFace SiglipForImageClassification (lazy-loaded) |
| Word Model          | Custom BiLSTM v7                                       |
| Environment         | Conda (openhands-api)                                  |

---

## Setup

### Backend

```bash
conda activate openhands-api
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Copy `.env.example` to `.env` and fill in your local paths:

```bash
cp .env.example .env
```

### Mobile App

```bash
cd mobile
# Point to your backend
echo "EXPO_PUBLIC_API_URL=http://YOUR_IP:8000" > .env
npx expo run:ios
```

Find your local IP:

```bash
ipconfig getifaddr en0
```

### Training

```bash
conda activate openhands-api
cd api/training

# Extract keypoints first
python extract_dataset.py asl100

# Train
python train.py                       # BiLSTM v7 — asl100
python train_300.py                   # BiLSTM v7 — asl300
python train_tgcn_pretrained_v2.py    # Best TGCN

# TGCN pretrained setup
python extract_att_matrices.py        # Run once before training TGCN pretrained
```

---

## License

WLASL dataset is licensed under the C-UDA license — **academic use only.**

---

## Acknowledgements

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Dongxu Li et al.
- [Voxel51/WLASL on HuggingFace](https://huggingface.co/datasets/Voxel51/WLASL)
- [MediaPipe](https://developers.google.com/mediapipe)
- [prithivMLmods/Alphabet-Sign-Language-Detection](https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection) — letter model

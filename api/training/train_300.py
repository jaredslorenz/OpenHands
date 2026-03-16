"""
BiLSTM v7 — asl300 — best: 54.3% val accuracy
"""

import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

KEYPOINTS_DIR = os.getenv("KEYPOINTS_DIR", "../WLASL/keypoints")
SPLIT_FILE    = os.getenv("SPLIT_FILE_300", "../WLASL/data/splits/asl300.json")
OUTPUT_MODEL  = os.getenv("OUTPUT_MODEL_300", "lstm_model_v7_300.pth")

NUM_CLASSES  = 300
NUM_FRAMES   = 50
NUM_JOINTS   = 55
IN_CHANNELS  = 4
EPOCHS       = 120
BATCH_SIZE   = 32
LR           = 1e-3
HIDDEN_SIZE  = 384
NUM_LAYERS   = 2
PATIENCE     = 15

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_splits():
    with open(os.path.join(KEYPOINTS_DIR, 'metadata.json')) as f:
        metadata = json.load(f)
    with open(SPLIT_FILE) as f:
        split_data = json.load(f)
    word_labels  = sorted(set(d['gloss'] for d in split_data))
    label_to_idx = {w:i for i,w in enumerate(word_labels)}
    valid_ids = {}
    for entry in split_data:
        for inst in entry['instances']:
            valid_ids[inst['video_id']] = label_to_idx[entry['gloss']]
    by_label = defaultdict(list)
    for vid in metadata:
        if vid not in valid_ids: continue
        npy = os.path.join(KEYPOINTS_DIR, f"{vid}.npy")
        if os.path.exists(npy):
            by_label[valid_ids[vid]].append((npy, valid_ids[vid]))
    train, val = [], []
    for label, samples in by_label.items():
        random.shuffle(samples)
        n_val = max(1, int(len(samples)*0.2))
        val.extend(samples[:n_val])
        train.extend(samples[n_val:])
    print(f"Train: {len(train)} | Val: {len(val)} | Classes: {len(by_label)}")
    return train, val


def compute_velocity(frames):
    vel = frames[1:] - frames[:-1]
    vel = torch.cat([vel, vel[-1:].clone()], dim=0)
    return torch.cat([frames, vel], dim=-1)

def rotate_frames(frames, max_deg=10):
    theta = np.radians(random.uniform(-max_deg, max_deg))
    rot   = torch.tensor([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]], dtype=frames.dtype)
    return frames @ rot.T

def time_warp(frames, sigma=0.15):
    T    = frames.shape[0]
    warp = torch.randn(T) * sigma
    warp = warp.cumsum(0)
    warp = warp - warp[0]
    warp = warp / (warp[-1] + 1e-6) * (T - 1)
    return frames[warp.clamp(0, T-1).long()]

def joint_dropout(frames, p=0.1):
    mask = (torch.rand(NUM_JOINTS) > p).float()
    return frames * mask.unsqueeze(0).unsqueeze(-1)

def cutout(frames, num_cuts=2, cut_len=4):
    frames = frames.clone()
    T = frames.shape[0]
    for _ in range(num_cuts):
        start = random.randint(0, T - cut_len)
        frames[start:start+cut_len] = 0
    return frames

def augment_frames(frames):
    frames += torch.randn_like(frames) * 0.01
    shift   = random.randint(-2, 2)
    if shift != 0:
        frames = torch.cat([frames[max(0,shift):], frames[:max(0,shift)]], dim=0)
    frames *= random.uniform(0.95, 1.05)
    frames  = rotate_frames(frames, max_deg=10)
    if random.random() < 0.5: frames = time_warp(frames)
    if random.random() < 0.4: frames = joint_dropout(frames)
    if random.random() < 0.4: frames = cutout(frames)
    if random.random() < 0.3:
        mask   = torch.rand(frames.shape) > 0.05
        frames = frames * mask.float()
    return frames


class ASLDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data   = torch.FloatTensor(np.load(path))
        frames = torch.zeros(NUM_FRAMES, NUM_JOINTS, 2)
        for t in range(NUM_FRAMES):
            frames[t, :, 0] = data[:, t*2]
            frames[t, :, 1] = data[:, t*2+1]
        if self.augment:
            frames = augment_frames(frames)
        return compute_velocity(frames).reshape(NUM_FRAMES, NUM_JOINTS*4), label


def mixup_batch(frames, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(frames.size(0), device=frames.device)
    return lam*frames + (1-lam)*frames[idx], labels, labels[idx], lam

def mixup_criterion(criterion, out, la, lb, lam):
    return lam * criterion(out, la) + (1-lam) * criterion(out, lb)


class ASLClassifier(nn.Module):
    def __init__(self, input_size=NUM_JOINTS*IN_CHANNELS,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.attn = nn.Linear(hidden_size*2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size*2), nn.Dropout(dropout),
            nn.Linear(hidden_size*2, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        return self.classifier((out * attn_w).sum(dim=1))


train_samples, val_samples = load_splits()
train_loader = DataLoader(ASLDataset(train_samples, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(ASLDataset(val_samples, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model     = ASLClassifier(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = OneCycleLR(optimizer, max_lr=LR,
                       steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val, no_improve = 0.0, 0
print(f"\nTraining up to {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS+1):
    model.train()
    t_loss=t_cor=t_tot=0
    for frames, labels in train_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if random.random() < 0.5:
            frames, la, lb, lam = mixup_batch(frames, labels)
            out     = model(frames)
            loss    = mixup_criterion(criterion, out, la, lb, lam)
            preds   = out.argmax(1)
            correct = (lam*(preds==la).float() + (1-lam)*(preds==lb).float()).sum().item()
        else:
            out     = model(frames)
            loss    = criterion(out, labels)
            correct = (out.argmax(1)==labels).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        t_loss += loss.item(); t_cor += correct; t_tot += len(labels)

    model.eval()
    v_cor=v_tot=0
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            v_cor += (model(frames).argmax(1)==labels).sum().item()
            v_tot += len(labels)

    t_acc = 100*t_cor/t_tot
    v_acc = 100*v_cor/v_tot
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss {t_loss/len(train_loader):.4f} | Train {t_acc:.1f}% | Val {v_acc:.1f}%")

    if v_acc > best_val:
        best_val, no_improve = v_acc, 0
        torch.save({'model_state_dict': model.state_dict(),
                    'val_acc': best_val, 'epoch': epoch,
                    'num_classes': NUM_CLASSES}, OUTPUT_MODEL)
        print(f"           ↑ Best! Saved.")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nDone. Best val: {best_val:.1f}%")
print(f"Saved to: {OUTPUT_MODEL}")
"""
Hybrid Spatial-GCN + BiLSTM for ASL recognition
Architecture:
  1. Spatial GCN per frame  — learns joint relationships within each frame
  2. BiLSTM across frames   — learns temporal signing patterns
  3. Attention pooling      — weights important frames
  4. Classifier             — 100 word prediction

Combines TGCN's spatial inductive bias with BiLSTM's proven temporal modeling.
Target: beat BiLSTM v7's 69.2% on asl100
"""

import json, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import GCNConv
from dotenv import load_dotenv

load_dotenv()

KEYPOINTS_DIR = os.getenv("KEYPOINTS_DIR", "../WLASL/keypoints")
SPLIT_FILE    = os.getenv("SPLIT_FILE_100", "../WLASL/data/splits/asl100.json")
OUTPUT_MODEL  = os.getenv("OUTPUT_MODEL_HYBRID", "lstm_tgcn_hybrid.pth")

NUM_CLASSES  = 100
NUM_FRAMES   = 50
NUM_JOINTS   = 55
IN_CHANNELS  = 4   # x, y, dx, dy — same as BiLSTM v7
GCN_HIDDEN   = 128 # spatial GCN output per joint
LSTM_HIDDEN  = 384 # BiLSTM hidden size — same as v7
LSTM_LAYERS  = 2
EPOCHS       = 120
BATCH_SIZE   = 32
LR           = 1e-3
PATIENCE     = 15
SEED         = 42

DEVICE = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"Device: {DEVICE}")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Skeleton Graph (spatial only — no temporal edges needed) ───────────────────
BODY_EDGES = [
    (0,1),(0,9),(0,10),(9,11),(10,12),
    (1,2),(1,5),(1,8),(2,3),(3,4),(5,6),(6,7),
]
def hand_edges(o):
    e = []
    for fs in [1,5,9,13,17]:
        e.append((o, o+fs))
        for j in range(fs, fs+3): e.append((o+j, o+j+1))
    return e

ALL_EDGES = (
    BODY_EDGES + hand_edges(13) + hand_edges(34)
    + [(7,13),(4,34),(13,34)]
)

def build_spatial_ei():
    src, dst = [], []
    for a, b in ALL_EDGES:
        src += [a, b]; dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long)

SPATIAL_EI = build_spatial_ei()
print(f"Spatial edges: {SPATIAL_EI.shape[1]//2}")

def batch_spatial_ei(ei, B_T, N):
    """Replicate edge index for B*T frames processed simultaneously."""
    return torch.cat([ei + i*N for i in range(B_T)], dim=1)


# ── Feature Engineering ────────────────────────────────────────────────────────
def compute_velocity(frames):
    """frames: (T, N, 2) → (T, N, 4) with velocity appended."""
    vel = frames[1:] - frames[:-1]
    vel = torch.cat([vel, vel[-1:].clone()], dim=0)
    return torch.cat([frames, vel], dim=-1)


# ── Augmentation — full v7 stack ───────────────────────────────────────────────
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
    """frames: (T, N, 2) raw coords."""
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


# ── Dataset ────────────────────────────────────────────────────────────────────
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


class ASLDataset(Dataset):
    """
    Returns:
        frames_4d : (T, N, 4)  — x, y, dx, dy per joint per frame
        label     : int
    """
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
        # Append velocity: (T, N, 2) → (T, N, 4)
        return compute_velocity(frames), label


# ── Mixup ──────────────────────────────────────────────────────────────────────
def mixup_batch(frames, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(frames.size(0), device=frames.device)
    return lam*frames + (1-lam)*frames[idx], labels, labels[idx], lam

def mixup_criterion(criterion, out, la, lb, lam):
    return lam * criterion(out, la) + (1-lam) * criterion(out, lb)


# ── Model ──────────────────────────────────────────────────────────────────────
class SpatialGCNBlock(nn.Module):
    """Single spatial GCN block with residual connection."""
    def __init__(self, in_c, out_c, dropout=0.3):
        super().__init__()
        self.gcn  = GCNConv(in_c, out_c)
        self.bn   = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, ei):
        return self.drop(F.gelu(self.bn(self.gcn(x, ei)) + self.res(x)))


class HybridGCNLSTM(nn.Module):
    """
    Spatial GCN encodes each frame's joint graph into a frame embedding,
    then BiLSTM models the temporal sequence of those embeddings.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()

        # ── Spatial GCN ──────────────────────────────────────────────────────
        # Processes all frames in parallel: (B*T, N, IN_CHANNELS) → (B*T, N, GCN_HIDDEN)
        self.spatial = nn.ModuleList([
            SpatialGCNBlock(IN_CHANNELS, 64,         dropout=0.3),
            SpatialGCNBlock(64,          GCN_HIDDEN,  dropout=0.3),
        ])

        # Pool joints → single frame embedding: (B*T, N, GCN_HIDDEN) → (B*T, GCN_HIDDEN)
        self.joint_attn = nn.Linear(GCN_HIDDEN, 1)

        # ── BiLSTM ───────────────────────────────────────────────────────────
        # Input: (B, T, GCN_HIDDEN) — sequence of spatial frame embeddings
        self.lstm = nn.LSTM(
            input_size=GCN_HIDDEN,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Temporal attention pooling
        self.temporal_attn = nn.Linear(LSTM_HIDDEN*2, 1)

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(LSTM_HIDDEN*2),
            nn.Dropout(dropout),
            nn.Linear(LSTM_HIDDEN*2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, s_ei):
        """
        x    : (B, T, N, 4)  — batch of frame sequences
        s_ei : (2, E)        — spatial edge index for one frame
        """
        B, T, N, C = x.shape

        # ── Spatial GCN (all B*T frames at once) ─────────────────────────────
        x_flat = x.reshape(B*T*N, C)
        ei     = batch_spatial_ei(s_ei, B*T, N).to(x.device)
        for block in self.spatial:
            x_flat = block(x_flat, ei)                  # (B*T*N, GCN_HIDDEN)

        # Attention pool over joints → frame embeddings
        x_frames = x_flat.reshape(B*T, N, GCN_HIDDEN)
        j_attn   = torch.softmax(self.joint_attn(x_frames), dim=1)  # (B*T, N, 1)
        x_frames = (x_frames * j_attn).sum(dim=1)                   # (B*T, GCN_HIDDEN)
        x_frames = x_frames.reshape(B, T, GCN_HIDDEN)               # (B, T, GCN_HIDDEN)

        # ── BiLSTM over frame sequence ────────────────────────────────────────
        lstm_out, _ = self.lstm(x_frames)                            # (B, T, LSTM_HIDDEN*2)

        # Temporal attention pooling
        t_attn  = torch.softmax(self.temporal_attn(lstm_out), dim=1) # (B, T, 1)
        pooled  = (lstm_out * t_attn).sum(dim=1)                     # (B, LSTM_HIDDEN*2)

        return self.classifier(pooled)


# ── Training ───────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
train_samples, val_samples = load_splits()

train_loader = DataLoader(ASLDataset(train_samples, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(ASLDataset(val_samples, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

s_ei  = SPATIAL_EI.to(DEVICE)
model = HybridGCNLSTM(NUM_CLASSES).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"BiLSTM v7 baseline: 69.2%\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = OneCycleLR(optimizer, max_lr=LR,
                       steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val, no_improve = 0.0, 0
print(f"Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS+1):
    model.train()
    t_loss=t_cor=t_tot=0
    for frames, labels in train_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if random.random() < 0.5:
            frames, la, lb, lam = mixup_batch(frames, labels)
            out     = model(frames, s_ei)
            loss    = mixup_criterion(criterion, out, la, lb, lam)
            preds   = out.argmax(1)
            correct = (lam*(preds==la).float() + (1-lam)*(preds==lb).float()).sum().item()
        else:
            out     = model(frames, s_ei)
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
            out = model(frames, s_ei)
            v_cor += (out.argmax(1)==labels).sum().item()
            v_tot += len(labels)

    t_acc = 100*t_cor/t_tot
    v_acc = 100*v_cor/v_tot
    print(f"Epoch {epoch:3d}/{EPOCHS} | Loss {t_loss/len(train_loader):.4f} | Train {t_acc:.1f}% | Val {v_acc:.1f}%")

    if v_acc > best_val:
        best_val, no_improve = v_acc, 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc':          best_val,
            'epoch':            epoch,
            'num_classes':      NUM_CLASSES,
            'spatial_ei':       SPATIAL_EI,
        }, OUTPUT_MODEL)
        print(f"           ↑ Best! Saved.")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nDone. Best val: {best_val:.1f}%")
print(f"BiLSTM v7 baseline: 69.2%")
print(f"Saved to: {OUTPUT_MODEL}")
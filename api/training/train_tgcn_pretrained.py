"""
GCN_muti_att with pretrained adjacency matrix initialization.

Strategy (Option B):
- Reconstruct GCN_muti_att exactly matching the WLASL authors' architecture
- Initialize ONLY the att matrices from the pretrained checkpoint
- All other weights (linear transforms, BN) initialize randomly
- Train from scratch on our MediaPipe keypoints with full v7 augmentation

Why this works:
- att matrices are 55x55 learned joint relationships — structural knowledge
  that encodes which joints co-vary for ASL signs
- This knowledge transfers regardless of coordinate system (OpenPose vs MediaPipe)
- Linear weights are coordinate-sensitive so we don't transfer those
- We get a meaningful head start on joint relationship structure vs random init

Target: beat ST-GCN v2's 16.4% on asl100
"""

import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from dotenv import load_dotenv

load_dotenv()

KEYPOINTS_DIR  = os.getenv("KEYPOINTS_DIR",  "../WLASL/keypoints")
SPLIT_FILE     = os.getenv("SPLIT_FILE_100", "../WLASL/data/splits/asl100.json")
OUTPUT_MODEL   = os.getenv("OUTPUT_MODEL_TGCN_PRETRAINED", "tgcn_pretrained.pth")
ATT_MATRICES   = os.getenv("ATT_MATRICES_PATH", "att_matrices.pt")

NUM_CLASSES  = 100
NUM_FRAMES   = 50
NUM_JOINTS   = 55
HIDDEN_SIZE  = 64    # must match pretrained checkpoint
NUM_STAGES   = 20   # must match pretrained checkpoint
INPUT_FEAT   = NUM_FRAMES * 2  # 100 — (x,y) per frame, joints as nodes
EPOCHS       = 120
BATCH_SIZE   = 64   # match original paper config
LR           = 1e-3
PATIENCE     = 15
SEED         = 42

DEVICE = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"Device: {DEVICE}")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Model — exact replica of WLASL GCN_muti_att ───────────────────────────────
import math

class GraphConvolution_att(nn.Module):
    """
    GCN layer with learned 55x55 adjacency matrix (att).
    Instead of a fixed graph, the model learns which joints relate to which.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att    = Parameter(torch.FloatTensor(55, 55))  # learned adjacency
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)   # (B, 55, out)
        output  = torch.matmul(self.att, support) # (B, 55, out)
        if self.bias is not None:
            return output + self.bias
        return output


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, is_resi=True):
        super().__init__()
        self.is_resi = is_resi
        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)
        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)
        self.do  = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        return y + x if self.is_resi else y


class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout,
                 num_stage=20, is_resi=True):
        super().__init__()
        self.num_stage = num_stage
        self.gc1  = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1  = nn.BatchNorm1d(55 * hidden_feature)
        self.gcbs = nn.ModuleList([
            GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi)
            for _ in range(num_stage)
        ])
        self.do   = nn.Dropout(p_dropout)
        self.act  = nn.Tanh()
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        # x: (B, 55, INPUT_FEAT)
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        for gcb in self.gcbs:
            y = gcb(y)
        out = torch.mean(y, dim=1)  # pool over joints
        return self.fc_out(out)


def load_pretrained_att(model, att_path):
    """
    Load only the att matrices from the pretrained checkpoint.
    All other weights stay as random initialization.
    """
    if not os.path.exists(att_path):
        print(f"WARNING: att_matrices.pt not found at {att_path}")
        print("Run extract_att_matrices.py first. Training with random init.")
        return model

    att_matrices = torch.load(att_path, map_location='cpu')
    model_state  = model.state_dict()

    loaded = 0
    for key, val in att_matrices.items():
        if key in model_state and model_state[key].shape == val.shape:
            model_state[key] = val
            loaded += 1
        else:
            print(f"  Skipping {key} — shape mismatch or not found")

    model.load_state_dict(model_state)
    print(f"Loaded {loaded}/{len(att_matrices)} att matrices from pretrained checkpoint")
    return model


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
    Returns frames in GCN_muti_att format: (55, NUM_FRAMES*2)
    Joints as nodes, frames*coords as features — matching the original paper.
    """
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data   = torch.FloatTensor(np.load(path))  # (55, 100) — already in this format
        frames = torch.zeros(NUM_FRAMES, NUM_JOINTS, 2)
        for t in range(NUM_FRAMES):
            frames[t, :, 0] = data[:, t*2]
            frames[t, :, 1] = data[:, t*2+1]

        if self.augment:
            frames = augment_frames(frames)

        # Reshape back to (55, 100) — joints × (frames*2) for GCN input
        return frames.permute(1, 0, 2).reshape(NUM_JOINTS, NUM_FRAMES*2), label


# ── Mixup ──────────────────────────────────────────────────────────────────────
def mixup_batch(frames, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(frames.size(0), device=frames.device)
    return lam*frames + (1-lam)*frames[idx], labels, labels[idx], lam

def mixup_criterion(criterion, out, la, lb, lam):
    return lam * criterion(out, la) + (1-lam) * criterion(out, lb)


# ── Training ───────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
train_samples, val_samples = load_splits()

train_loader = DataLoader(ASLDataset(train_samples, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(ASLDataset(val_samples, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Build model and load pretrained att matrices
model = GCN_muti_att(
    input_feature=INPUT_FEAT,
    hidden_feature=HIDDEN_SIZE,
    num_class=NUM_CLASSES,
    p_dropout=0.3,
    num_stage=NUM_STAGES,
).to(DEVICE)

model = load_pretrained_att(model, ATT_MATRICES)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Previous TGCN best: 16.4% | BiLSTM v7 baseline: 69.2%\n")

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
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc':          best_val,
            'epoch':            epoch,
            'num_classes':      NUM_CLASSES,
        }, OUTPUT_MODEL)
        print(f"           ↑ Best! Saved.")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

print(f"\nDone. Best val: {best_val:.1f}%")
print(f"Previous TGCN best: 16.4%")
print(f"BiLSTM v7 baseline: 69.2%")
print(f"Saved to: {OUTPUT_MODEL}")
"""
Extract learned adjacency (att) matrices from the WLASL pretrained TGCN checkpoint.
These 55x55 matrices encode which joints relate to which for ASL signing —
structural knowledge that transfers regardless of coordinate system.

Usage:
    python extract_att_matrices.py

Saves:
    att_matrices.pt — dict of all att matrices, ready to load in train_tgcn_pretrained.py
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

CKPT_PATH  = os.getenv("PRETRAINED_ATT", "../WLASL/code/TGCN/archived/asl100/ckpt.pth")
SAVE_PATH  = os.getenv("ATT_MATRICES_PATH", "att_matrices.pt")

print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location='cpu')

# Extract all att matrices
att_matrices = {}
for key, val in ckpt.items():
    if 'att' in key:
        att_matrices[key] = val
        print(f"  {key}: {val.shape} | min={val.min():.4f} max={val.max():.4f} mean={val.mean():.4f}")

print(f"\nExtracted {len(att_matrices)} att matrices")

# Verify shapes — all should be (55, 55)
shapes = set(v.shape for v in att_matrices.values())
assert shapes == {torch.Size([55, 55])}, f"Unexpected shapes: {shapes}"
print("All matrices are (55, 55) ✓")

# Save
torch.save(att_matrices, SAVE_PATH)
print(f"\nSaved to: {SAVE_PATH}")
print("Ready for train_tgcn_pretrained.py")
"""
keypoints.py — feature engineering helpers for inference.
Velocity computation, input preparation, and TTA augmentation
are kept here so inference.py stays focused on prediction logic.
"""

import numpy as np
import torch

from config import NUM_FRAMES, NUM_JOINTS, IN_CHANNELS


def compute_velocity(frames: torch.Tensor) -> torch.Tensor:
    """
    Append per-joint velocity to raw coordinates.

    Args:
        frames: (T, N, 2) — raw x, y coordinates
    Returns:
        (T, N, 4) — x, y, dx, dy
    """
    vel = frames[1:] - frames[:-1]
    vel = torch.cat([vel, vel[-1:].clone()], dim=0)
    return torch.cat([frames, vel], dim=-1)


def prepare_input(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert raw keypoint tensor to model-ready input.

    Args:
        tensor: (1, 55, 100) — output of extract_keypoints_from_video
    Returns:
        (1, 50, 220) — batch-ready model input
    """
    t      = tensor.squeeze(0)                          # (55, 100)
    frames = torch.zeros(NUM_FRAMES, NUM_JOINTS, 2)
    for i in range(NUM_FRAMES):
        frames[i, :, 0] = t[:, i * 2]
        frames[i, :, 1] = t[:, i * 2 + 1]
    frames = compute_velocity(frames)                   # (50, 55, 4)
    return frames.reshape(1, NUM_FRAMES, NUM_JOINTS * IN_CHANNELS)


def tta_augment(frames: torch.Tensor) -> torch.Tensor:
    """
    Apply a single randomised augmentation pass for test-time augmentation.
    Mirrors the training augmentation stack at reduced intensity.

    Args:
        frames: (1, 50, 220) — prepared model input
    Returns:
        (1, 50, 220) — augmented variant
    """
    f  = frames.clone().reshape(NUM_FRAMES, NUM_JOINTS, IN_CHANNELS)
    xy = f[:, :, :2]

    # Scale jitter
    xy = xy * torch.empty(1).uniform_(0.95, 1.05).item()

    # Rotation
    theta = np.radians(np.random.uniform(-8, 8))
    rot   = torch.tensor(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]],
        dtype=xy.dtype,
    )
    xy = xy @ rot.T

    # Gaussian noise
    xy = xy + torch.randn_like(xy) * 0.008

    # Recompute velocity from augmented coordinates
    vel_new = torch.cat(
        [xy[1:] - xy[:-1], (xy[-1:] - xy[-2:-1]).clone()], dim=0
    )
    f_aug = torch.cat([xy, vel_new], dim=-1)
    return f_aug.reshape(1, NUM_FRAMES, NUM_JOINTS * IN_CHANNELS)
"""
inference.py — prediction logic for letter and word endpoints.
All ML inference lives here; HTTP concerns stay in main.py.
"""

import io
import torch
from PIL import Image, ImageOps

import config
from keypoints import prepare_input, tta_augment


def predict_letter_from_image(contents: bytes) -> dict:
    """
    Predict a single ASL letter from an image.

    Args:
        contents: raw image bytes
    Returns:
        {"letter": str, "confidence": float}
    """
    letter_model, letter_processor = config.get_letter_model()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    # Crop to the region where the signing hand typically appears
    w, h  = image.size
    image = image.crop((int(w * 0.1), int(h * 0.25), int(w * 0.9), int(h * 0.65)))

    inputs = letter_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        probs = torch.nn.functional.softmax(
            letter_model(**inputs).logits, dim=1
        ).squeeze()

    pred_idx = int(torch.argmax(probs))
    return {
        "letter":     config.LETTER_LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
    }


def predict_word_from_tensor(tensor: torch.Tensor) -> dict:
    """
    Predict an ASL word from a keypoint tensor using TTA.

    Args:
        tensor: (1, 55, 100) — output of extract_keypoints_from_video
    Returns:
        {"word": str, "confidence": float, "top5": list[dict]}
    """
    base_input = prepare_input(tensor)

    with torch.no_grad():
        # Base pass + TTA_RUNS augmented passes, averaged
        all_probs = torch.softmax(config.word_model(base_input), dim=1)
        for _ in range(config.TTA_RUNS):
            all_probs += torch.softmax(
                config.word_model(tta_augment(base_input)), dim=1
            )

    probs      = (all_probs / (config.TTA_RUNS + 1)).squeeze()
    pred_idx   = int(torch.argmax(probs))
    confidence = float(probs[pred_idx])

    top5 = torch.topk(probs, 5)

    # Log prediction summary to server stdout
    print("\n── Word Prediction (TTA) ────────────────")
    for rank, (idx, prob) in enumerate(zip(top5.indices, top5.values)):
        marker = " ← predicted" if rank == 0 else ""
        print(f"  {rank+1}. {config.word_labels[int(idx)]:<20} {float(prob)*100:.1f}%{marker}")
    print(f"  (averaged over {config.TTA_RUNS + 1} passes)")
    print("─────────────────────────────────────────\n")

    return {
        "word":       config.word_labels[pred_idx],
        "confidence": confidence,
        "top5": [
            {"word": config.word_labels[int(idx)], "confidence": float(prob)}
            for idx, prob in zip(top5.indices, top5.values)
        ],
    }
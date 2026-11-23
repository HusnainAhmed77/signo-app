"""MobileNetV3-Small image inference for the Sign-O desktop app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top3: tuple[tuple[str, float], ...]
    character: str | None


def prepare_image(image: Image.Image) -> Image.Image:
    """Apply the training dataset's RGB and centered white-padding policy."""
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(
        rgb_image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def load_label_mapping(path: Path) -> dict[str, str]:
    """Load exact dataset-label-to-Urdu mappings supplied by the project owner."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Label mapping file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label mapping is not valid JSON: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError("Label mapping must be a JSON object of class names to Urdu characters.")
    if any(not isinstance(key, str) or not isinstance(char, str) for key, char in value.items()):
        raise ValueError("Every label mapping key and value must be a string.")
    return value


class MobileNetSignPredictor:
    """Loads a trained MobileNetV3-Small checkpoint and predicts a still image."""

    def __init__(self, checkpoint_path: Path, label_mapping_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.label_mapping = load_label_mapping(label_mapping_path)
        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None
        self.class_names: tuple[str, ...] = ()
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and torchvision are required. Install the app dependencies "
                "in the selected Python environment."
            ) from exc

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "MobileNet transfer-learning checkpoint is missing. Copy "
                f"best_model_tl.pt from Drive to {self.checkpoint_path}."
            )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Could not load model checkpoint: {exc}") from exc

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a model metadata dictionary.")
        if checkpoint.get("architecture") != "MobileNetV3-Small-TL":
            raise ValueError("Checkpoint is not the expected MobileNetV3-Small transfer model.")
        raw_class_names = checkpoint.get("class_names")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(raw_class_names, (list, tuple)) or not raw_class_names:
            raise ValueError("Checkpoint is missing its ordered class_names list.")
        if any(not isinstance(name, str) for name in raw_class_names):
            raise ValueError("Checkpoint class_names must contain only strings.")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint is missing model_state_dict.")
        if tuple(checkpoint.get("image_size", ())) != IMAGE_SIZE:
            raise ValueError(f"Expected a {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} checkpoint.")

        class_names = tuple(raw_class_names)
        missing_labels = set(class_names) - set(self.label_mapping)
        if missing_labels:
            raise ValueError(
                "Urdu label mapping is missing model classes: "
                + ", ".join(sorted(missing_labels))
            )
        empty_labels = [name for name in class_names if not self.label_mapping[name].strip()]
        if empty_labels:
            raise ValueError(
                "Urdu label mapping has empty values for: " + ", ".join(sorted(empty_labels))
            )
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Linear(in_features, len(class_names)))
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint weights do not match MobileNetV3-Small: {exc}") from exc
        model.to(device)
        model.eval()

        self._torch = torch
        self._device = device
        self._model = model
        self.class_names = class_names

    def predict_rgb(self, image: Image.Image) -> Prediction:
        """Return the top class and probabilities for an RGB/PIL still image."""
        torch = self._torch
        prepared = prepare_image(image)
        tensor = torch.from_numpy(
            np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1).copy()
        ).div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0].cpu()
        top_count = min(3, len(self.class_names))
        values, indices = torch.topk(probabilities, top_count)
        top3 = tuple(
            (self.class_names[int(index)], float(value))
            for value, index in zip(values, indices)
        )
        label, confidence = top3[0]
        mapped_character = self.label_mapping.get(label, "").strip() or None
        return Prediction(label, confidence, top3, mapped_character)

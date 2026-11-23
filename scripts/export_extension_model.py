"""Export the app's MobileNetV3-Small checkpoint to ONNX for the Chrome extension.

Usage (from the project root, in the desktop-app environment):
    python scripts/export_extension_model.py

Writes into signo-chrome-extension/models/:
    best_model_tl.onnx   - float32 model, input "input" [1, 3, 224, 224] (ImageNet-normalized RGB),
                           output "logits" [1, num_classes]
    classes.json         - ordered class names from the checkpoint plus their Urdu characters
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signo_app.predictor import IMAGE_SIZE, MobileNetSignPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "ualpha40_transfer_learning_v1" / "best_model_tl.pt",
    )
    parser.add_argument("--labels", type=Path, default=ROOT / "signo_app" / "labels.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "signo-chrome-extension" / "models")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    # Reuse the app loader so the export gets exactly the same checkpoint validation.
    predictor = MobileNetSignPredictor(args.checkpoint, args.labels)
    torch = predictor._torch
    model = predictor._model.to("cpu").eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "best_model_tl.onnx"
    dummy = torch.zeros(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    classes = [
        {"label": name, "character": predictor.label_mapping[name].strip()}
        for name in predictor.class_names
    ]
    classes_path = args.out_dir / "classes.json"
    classes_path.write_text(
        json.dumps({"image_size": list(IMAGE_SIZE), "classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {classes_path} ({len(classes)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signo_model import compare_pretrained as benchmark


@pytest.mark.parametrize("model_name", benchmark.SUPPORTED_MODELS)
def test_build_model_replaces_classifier_for_all_classes(model_name):
    model = benchmark.build_model(model_name, class_count=36, pretrained=False)
    sample = torch.zeros(1, 3, *benchmark.IMAGE_SIZE)

    model.eval()
    with torch.inference_mode():
        output = model(sample)

    assert output.shape == (1, 36)


def test_macro_metrics_uses_all_class_indices():
    accuracy, precision, recall, f1, confusion = benchmark._macro_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        class_count=2,
    )

    assert accuracy == pytest.approx(0.75)
    assert precision == pytest.approx((1 + 2 / 3) / 2)
    assert recall == pytest.approx(0.75)
    assert f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert confusion.tolist() == [[1, 1], [0, 2]]


def test_train_and_evaluate_writes_checkpoint_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "OUTPUT_ROOT", tmp_path / "models" / "comparison")
    rows = []
    for split in ("train", "validation", "test"):
        for label in ("Alif", "Bay"):
            image_path = tmp_path / f"{split}_{label}.png"
            color = (255, 255, 255) if label == "Alif" else (0, 0, 0)
            from PIL import Image

            Image.new("RGB", benchmark.IMAGE_SIZE, color).save(image_path)
            rows.append(
                {
                    "label": label,
                    "processed_image_path": str(image_path),
                    "split": split,
                    "source_sha256": f"{split}_{label}",
                }
            )

    result = benchmark.train_and_evaluate(
        "mobilenet_v3_small",
        rows,
        ["Alif", "Bay"],
        epochs=1,
        patience=1,
        batch_size=2,
        seed=42,
        pretrained=False,
    )
    output_dir = benchmark.OUTPUT_ROOT / "mobilenet_v3_small"

    assert result["test_image_count"] == 2
    assert result["pretrained"] is False
    assert (output_dir / "best_checkpoint.pt").is_file()
    assert (output_dir / "training_history.csv").is_file()
    with (output_dir / "classification_report.csv").open(newline="") as report_file:
        assert [row["label"] for row in csv.DictReader(report_file)] == ["Alif", "Bay"]

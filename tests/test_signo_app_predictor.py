import json
from pathlib import Path

import pytest
from PIL import Image

from signo_app.hand_tracking import landmarks_to_pixels
from signo_app.predictor import IMAGE_SIZE, load_label_mapping, prepare_image


def test_prepare_image_matches_training_canvas_policy():
    source = Image.new("RGB", (400, 200), (255, 0, 0))
    result = prepare_image(source)
    assert result.size == IMAGE_SIZE
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((112, 112)) == (255, 0, 0)


def test_prepare_image_converts_grayscale_to_rgb():
    result = prepare_image(Image.new("L", (200, 200), 128))
    assert result.mode == "RGB"
    assert result.size == IMAGE_SIZE


def test_landmarks_to_pixels_clamps_points_to_frame():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    points = landmarks_to_pixels(
        [Point(0.5, 0.25), Point(-0.2, 1.2)],
        width=200,
        height=100,
    )
    assert points == [(100, 25), (0, 99)]


def test_load_label_mapping_requires_json_object(tmp_path):
    mapping_file = tmp_path / "labels.json"
    mapping_file.write_text(json.dumps({"Alif": "ا"}), encoding="utf-8")
    assert load_label_mapping(mapping_file) == {"Alif": "ا"}

    mapping_file.write_text(json.dumps(["Alif", "ا"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_label_mapping(mapping_file)


def test_label_mapping_has_published_urdu_characters():
    mapping = load_label_mapping(Path(__file__).parents[1] / "signo_app" / "labels.json")
    assert len(mapping) == 39
    assert mapping["1-Hay"] == "ح"
    assert mapping["Kaf"] == "ق"
    assert mapping["Kiaf"] == "ک"
    assert mapping["Taay"] == "ٹ"
    assert mapping["Tuey"] == "ط"


def test_load_label_mapping_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Label mapping file not found"):
        load_label_mapping(tmp_path / "missing.json")

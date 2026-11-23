import csv

import numpy as np
import pytest
from PIL import Image

from signo_app.text_to_sign import (
    load_example_images,
    load_landmark_reference,
    split_urdu_text,
)


def test_split_urdu_text_maps_letters_and_preserves_spaces():
    mapping = {"Alif": "ا", "Bay": "ب"}

    steps = split_urdu_text("اب ب؟", mapping)

    assert [(step.character, step.label, step.is_space) for step in steps] == [
        ("ا", "Alif", False),
        ("ب", "Bay", False),
        (" ", None, True),
        ("ب", "Bay", False),
        ("؟", None, False),
    ]


def test_split_urdu_text_normalizes_unicode():
    steps = split_urdu_text("ا\u0654", {"Alif": "أ"})

    assert [(step.character, step.label) for step in steps] == [("أ", "Alif")]


def test_split_urdu_text_maps_common_urdu_and_arabic_forms_to_dataset_classes():
    mapping = {"Alif": "ا", "Cyeh": "ى", "Kiaf": "ک"}

    steps = split_urdu_text("آ أ إ ٱ ي ی ك", mapping)

    assert [(step.character, step.label) for step in steps if not step.is_space] == [
        ("آ", "Alif"),
        ("أ", "Alif"),
        ("إ", "Alif"),
        ("ٱ", "Alif"),
        ("ي", "Cyeh"),
        ("ی", "Cyeh"),
        ("ك", "Kiaf"),
    ]


def test_split_urdu_text_does_not_guess_a_different_dataset_letter():
    steps = split_urdu_text("ہ ھ", {"1-Hay": "ح", "Dochahay": "ھ"})

    assert [(step.character, step.label) for step in steps] == [
        ("ہ", None),
        (" ", None),
        ("ھ", "Dochahay"),
    ]


def test_split_urdu_text_maps_jeem_and_rre_to_landmark_classes():
    steps = split_urdu_text("ج ڑ", {})

    assert [(step.character, step.label) for step in steps] == [
        ("ج", "Jeem"),
        (" ", None),
        ("ڑ", "rray"),
    ]


def test_split_urdu_text_collapses_whitespace_to_one_word_break():
    steps = split_urdu_text(" ا \t ب ", {"Alif": "ا", "Bay": "ب"})

    assert [step.character for step in steps] == [" ", "ا", " ", "ب", " "]


def test_load_example_images_uses_existing_manifest_files(tmp_path):
    image_path = tmp_path / "data" / "example.png"
    image_path.parent.mkdir()
    image_path.touch()
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["label", "processed_image_path"],
        )
        writer.writeheader()
        writer.writerow({"label": "Alif", "processed_image_path": "data/example.png"})

    assert load_example_images(manifest_path, tmp_path) == {"Alif": image_path}


def test_load_example_images_reports_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepare the public dataset"):
        load_example_images(tmp_path / "missing.csv", tmp_path)


def test_load_landmark_reference_renders_a_representative_hand(tmp_path):
    class_directory = tmp_path / "Jeem"
    class_directory.mkdir()
    sequence = np.zeros((30, 225), dtype=np.float32)
    points = np.array(
        [
            [0.00, 0.00, 0.00], [-0.08, -0.10, 0], [-0.10, -0.20, 0],
            [-0.11, -0.30, 0], [-0.12, -0.40, 0], [-0.04, -0.12, 0],
            [-0.04, -0.22, 0], [-0.04, -0.32, 0], [-0.04, -0.42, 0],
            [0.00, -0.13, 0], [0.00, -0.24, 0], [0.00, -0.35, 0],
            [0.00, -0.46, 0], [0.04, -0.12, 0], [0.05, -0.22, 0],
            [0.06, -0.32, 0], [0.07, -0.41, 0], [0.08, -0.08, 0],
            [0.10, -0.15, 0], [0.12, -0.22, 0], [0.14, -0.28, 0],
        ],
        dtype=np.float32,
    )
    sequence[12, 63:126] = points.reshape(-1)
    np.save(class_directory / "Jeem_0.npy", sequence)

    image = load_landmark_reference(tmp_path, "Jeem")

    assert isinstance(image, Image.Image)
    assert image.size == (720, 520)
    assert image.getpixel((0, 0)) == (245, 242, 236)
    assert len(image.getcolors(maxcolors=1_000_000)) > 2


def test_load_landmark_reference_rejects_unapproved_class(tmp_path):
    with pytest.raises(ValueError, match="No approved Urdu character mapping"):
        load_landmark_reference(tmp_path, "Hain")

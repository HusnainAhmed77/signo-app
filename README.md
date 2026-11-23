# Sign-O

Sign-O is a project that recognizes **static Pakistan Sign Language handshapes** and shows the matching Urdu letter. The current desktop app can watch a camera or a picture, show hand-tracking visuals, and build Urdu text in a preview.

## What it can do now

- Show a live camera view with hand landmarks and an index-finger trail.
- Update the predicted class, Urdu letter, and confidence automatically while a hand is visible.
- Predict from a selected picture when a hand is detected.
- Add the displayed letter to an app-local Urdu text preview; space, backspace, clear, and copy controls are available.
- Enter Urdu text in **Text to signs** to step through its letters using alphabet-dataset reference images. The second dataset also supplies landmark visualizations for **ج** and **ڑ**; these are representative static hand-skeleton drawings, not photos or model predictions, and their letter mappings still need PSL-expert confirmation.

The interface uses warm paper backgrounds, charcoal text, restrained orange accents, and rounded cards. The desktop window adapts to its size: side-by-side panels stack on narrower windows, and each tab scrolls vertically when its content needs more room.

## Important limits

- Sign-O classifies **one still image at a time**. The hand trail is only a visual aid; the model does not recognize movement-based signs, words, or sentences.
- A prediction is displayed automatically, but it is **not** added to the text preview until you click **Add character to preview**.
- This is currently a desktop preview app, **not an installable system-wide keyboard**. It does not type into other applications.
- **Text to signs** spells text letter by letter; it does not translate words into natural PSL signs. Its pictures are still dataset examples, not animated instructions.
- The model is trained on dataset images. Its reported test results are image-level results and do not guarantee webcam accuracy.

## Run the desktop app

Use **Python 3.12**. Open a terminal in the project folder (the folder containing this `README.md`) and check Python:

```bash
python --version
```

The version command should report Python 3.12. Create and activate a virtual environment:

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the desktop app dependencies and launch Sign-O from the project folder:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
python -m signo_app
```

The app needs the model checkpoint at:

```text
models/ualpha40_transfer_learning_v1/best_model_tl.pt
```

The checkpoint and Urdu class mapping are checked when the app starts. If the checkpoint is missing or incompatible, the app displays an error instead of making predictions.

## Use Sign-O

1. Allow camera access if your operating system asks.
2. Click **Start camera** and show one hand clearly.
3. Read the predicted class and Urdu letter in the **Prediction** panel. They update automatically while a hand is visible.
4. Click **Add character to preview** when you want to keep the displayed letter in the text box.
5. Use **Space**, **Backspace**, **Clear**, or **Copy text** to edit or copy the preview.

To try a picture instead, click **Choose image**. A hand must be detected in the picture for a prediction to appear.

## Compare pretrained image models

The comparison runner fine-tunes four ImageNet-pretrained torchvision models on the same UAlpha40 classes and fixed train/validation/test image split: MobileNetV3-Small, MobileNetV3-Large, ResNet18, and EfficientNet-B0. It uses the same image size, training augmentation, optimizer, epoch limit, and validation macro-F1 early stopping for each architecture. The test partition is evaluated after training; compare validation macro-F1 when choosing a candidate, then use its test result as a final held-out check. These are image-level results, not signer-independent or webcam guarantees.

Run the existing training notebook through its split-creation section first if `data/processed/ualpha40_image_splits_v1/image_split_manifest.csv` is missing. Then, from the project root, run one or more architectures:

```bash
python -m signo_model.compare_pretrained --models mobilenet_v3_small resnet18 efficientnet_b0
```

Alternatively, open [02_compare_pretrained_models.ipynb](notebooks/02_compare_pretrained_models.ipynb) and run its cells yourself. The notebook's training cell is the only one that starts training.

The first run downloads the official ImageNet weights for the selected architectures and requires internet access. Omit `--models` to run all four, or adjust the training duration, for example `--epochs 30 --patience 7`. Results, checkpoints, training histories, per-class reports, and confusion matrices are written under `models/ualpha40_pretrained_comparison_v1/`. The comparison does not replace the app's current `best_model_tl.pt`; selecting a different production model would require a separate integration and validation step. This experiment compares image backbones only: it does not add letters missing from UAlpha40 or compare the separate 30-frame landmark dataset.

To see Urdu text spelled out, open **Text to signs**, type or paste text, then click **Show signs**. Use **Previous**, **Next**, **Play**, and **Pause** to review the sequence. Adjust **Time per letter** to control how quickly the examples change. Spaces appear as word breaks. Common Unicode forms of Alif, Yeh, and Kaf are linked to their dataset classes. The HF landmark dataset adds static visualizations for ج (`Jeem`) and ڑ (`rray`); those names have not yet been confirmed by a PSL expert. Other unsupported characters are identified rather than shown with a potentially incorrect sign. The webcam classifier remains limited to its trained 36 UAlpha40 image classes.

To hear the entered text, use **Speak Urdu text**. The speech-rate slider controls how quickly the operating system's Urdu voice reads it, and **Stop speaking** ends playback. Urdu speech depends on an Urdu voice being installed and supported by your system; Sign-O will say when one is unavailable rather than use a different language's voice.

## Run tests

With the project environment active, install pytest if it is not already available, then run:

```bash
python -m pip install pytest
python -m pytest -q
```

## Project folders

| Path | What it contains |
|---|---|
| `signo_app/` | Desktop interface, camera hand-tracking visuals, model inference, and Urdu label mapping |
| `signo_model/` | Dataset preprocessing and image-dataset utilities |
| `notebooks/` | Dataset preprocessing and image-classifier training notebooks |
| `data/` | Dataset and processed-data folders |
| `models/` | Model checkpoints and evaluation outputs |
| `requirements-app.txt` | Dependencies for running the desktop app |
| `requirements.txt` | Dependencies for model/data work and notebooks |

The app and model/notebook workflows have different dependencies. Install `requirements-app.txt` to run the desktop app; the larger `requirements.txt` is for data and model development.
If you work on the notebooks, use a separate virtual environment for `requirements.txt` so its dependencies do not conflict with the desktop app environment.

## Data and model notes

The image workflow uses public sign datasets and keeps preprocessing separate from app hand tracking. The text-to-sign viewer uses example images from `data/processed/ualpha40_images_v1/`, so that processed dataset must be present for the feature to display examples. See `notebooks/00_public_data_preprocessing.ipynb` for dataset preparation and `notebooks/01_train_image_classifier (1).ipynb` for model training and evaluation. Urdu characters for the model's 36 classes are listed in `signo_app/labels.json`.

The viewer spells words using alphabet examples; it does not claim to provide natural word-level PSL signs or animated speech. Urdu audio reads the entered text using the device's installed voice and does not change or animate the sign images. Check dataset licence terms before sharing or redistributing dataset images or derived data.

For the detailed desktop keyboard plan, see [PART2_App_and_UI.md](PART2_App_and_UI.md). For the model and training plan, see [PART1_FULL_PLAN_Model_Training (1).md](PART1_FULL_PLAN_Model_Training%20(1).md).

"""Sign-O desktop companion UI. Run with ``python -m signo_app``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .predictor import MobileNetSignPredictor, Prediction, load_label_mapping
from .hand_tracking import HandTracker
from .speech import SPEECH_RATE_MAX_PERCENT, SPEECH_RATE_MIN_PERCENT, speech_rate_from_percent
from .text_to_sign import (
    LANDMARK_CHARACTER_LABELS,
    TextSignStep,
    load_example_images,
    load_landmark_reference,
    split_urdu_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "ualpha40_transfer_learning_v1" / "best_model_tl.pt"
LABEL_MAPPING = Path(__file__).with_name("labels.json")
IMAGE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "ualpha40_images_v1" / "manifest.csv"
HF_LANDMARK_ROOT = PROJECT_ROOT / "data" / "external" / "hf_psl" / "data"
PREDICTION_INTERVAL_SECONDS = 0.4

# Confirmation method settings
SETTINGS_FILE = PROJECT_ROOT / "signo_app_settings.json"
DEFAULT_CONFIRM_METHOD = "mouse"  # mouse | pinch | dwell | both
DEFAULT_DWELL_SECONDS = 1.5
DEFAULT_CLICK_HAND = "left"  # left | right (the hand used to confirm)
DEFAULT_PINCH_THRESHOLD = 0.06
PINCH_COOLDOWN_SECONDS = 1.0


class BrandMark(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setToolTip("Sign-O")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().color(self.foregroundRole()))
        center = self.rect().center()
        painter.translate(center)
        for _ in range(4):
            painter.drawEllipse(-5, -16, 10, 12)
            painter.rotate(90)


class SignOWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sign-O | PSL Learning Studio")
        self.resize(1280, 900)
        self.setMinimumSize(760, 560)
        self.predictor: MobileNetSignPredictor | None = None
        self.last_prediction: Prediction | None = None
        self.camera = None
        self.hand_tracker: HandTracker | None = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.latest_rgb: Image.Image | None = None
        self.latest_has_hand = False
        self._last_prediction_time = 0.0
        self.sign_sequence_timer = QTimer(self)
        self.sign_sequence_timer.timeout.connect(self._advance_text_sign)
        self.sign_sequence: list[TextSignStep] = []
        self.sign_example_images: dict[str, Path] = {}
        self.sign_label_mapping: dict[str, str] = {}
        self.sign_sequence_index = 0
        self.sign_reference_pixmap: QPixmap | None = None
        self.landmark_reference_pixmaps: dict[str, QPixmap] = {}
        self.speech = None
        # Confirmation gesture state
        self.confirm_method = DEFAULT_CONFIRM_METHOD
        self.dwell_seconds = DEFAULT_DWELL_SECONDS
        self.click_hand = DEFAULT_CLICK_HAND
        self.pinch_threshold = DEFAULT_PINCH_THRESHOLD
        self._dwell_label: str | None = None
        self._dwell_start_time: float | None = None
        self._last_pinch_time: float = 0.0
        self._build_ui()
        self._load_predictor()
        self._load_settings()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(10)
        brand_mark = BrandMark()
        brand_mark.setObjectName("brandMark")
        header.addWidget(brand_mark)
        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title = QLabel("Sign-O")
        title.setObjectName("title")
        subtitle = QLabel("PSL Learning Studio · Pakistan Sign Language")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()
        self.keyboard_state = QLabel("PREVIEW MODE")
        self.keyboard_state.setObjectName("statusBadge")
        self.keyboard_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.keyboard_state, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.sign_tab, self.sign_page = self._scrollable_tab()
        self._build_sign_tab(self.sign_page)
        self.settings_tab, self.settings_page = self._scrollable_tab()
        self._build_settings_tab(self.settings_page)
        self.library_tab, self.library_page = self._scrollable_tab()
        self._build_library_tab(self.library_page)
        self.tabs.addTab(self.sign_tab, "Sign to text")
        self.tabs.addTab(self.library_tab, "Text to signs")
        self.tabs.addTab(self.settings_tab, "Settings")
        layout.addWidget(self.tabs, stretch=1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QWidget#appRoot { background: #f8f7f3; color: #171717; font-family: "Avenir Next", Arial; font-size: 14px; }
            QWidget { color: #262522; font-family: "Avenir Next", Arial; font-size: 14px; }
            QWidget#brandMark { color: #ff7a2f; }
            QLabel#title { font-size: 22px; font-weight: 600; color: #171717; letter-spacing: -0.5px; }
            QLabel#subtitle { color: #74716c; font-size: 10px; font-weight: 600; letter-spacing: 1.2px; }
            QLabel#statusBadge { background: #fff2e9; color: #b94f19; padding: 6px 12px; border: 1px solid #f3d9c8; border-radius: 12px; font-size: 9px; font-weight: 700; }
            QLabel#statusBadgeError { background: #fff0ed; color: #a63f2f; padding: 6px 12px; border: 1px solid #f0d2cb; border-radius: 12px; font-size: 9px; font-weight: 700; }
            QTabWidget::pane { background: transparent; border: 0; top: 2px; }
            QTabBar::tab { background: transparent; color: #74716c; padding: 8px 12px; margin-right: 4px; border: 0; border-bottom: 2px solid transparent; font-weight: 500; }
            QTabBar::tab:hover { color: #c95c21; }
            QTabBar::tab:selected { background: transparent; color: #171717; border-bottom: 2px solid #ff7a2f; font-weight: 700; }
            QComboBox { selection-background-color: #ff7a2f; color: #171717; padding: 5px; min-width: 80px; }
            QComboBox QAbstractItemView { background: white; color: #171717; selection-background-color: #ff7a2f; selection-color: #171717; }
            QScrollArea#pageScroll { background: transparent; border: 0; }
            QWidget#tabPage { background: #f8f7f3; }
            QScrollBar:vertical { width: 9px; background: transparent; margin: 3px 1px 3px 5px; }
            QScrollBar::handle:vertical { background: #d3cec5; min-height: 28px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #b8afa3; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            QFrame#card { background: #fff; border: 1px solid #e4e0d8; border-radius: 18px; }
            QLabel#sectionTitle { font-size: 16px; font-weight: 600; color: #242320; }
            QLabel#eyebrow { color: #91877d; font-size: 9px; font-weight: 700; letter-spacing: 1.5px; }
            QLabel#pageTitle { color: #171717; font-size: 24px; font-weight: 500; letter-spacing: -0.6px; }
            QLabel#pageDescription { color: #74716c; font-size: 13px; }
            QLabel#predictionLabel { font-size: 20px; font-weight: 600; color: #d95f22; }
            QLabel#characterLabel { font-size: 64px; font-weight: 700; color: #171717; }
            QLabel#currentLetter { background: #fff4eb; color: #d95f22; border: 1px solid #f4ddce; border-radius: 18px; font-size: 56px; font-weight: 600; }
            QLabel#cameraPreview { background: #22211f; color: #e2ded7; border: 1px solid #3b3935; border-radius: 14px; }
            QLabel#mutedText { color: #77736d; }
            QPushButton { background: #ff7a2f; color: white; border: 1px solid #ff7a2f; border-radius: 22px; padding: 10px 18px; min-height: 20px; font-weight: 600; }
            QPushButton:hover { background: #e96824; border-color: #e96824; }
            QPushButton:pressed { background: #c95418; }
            QPushButton:focus { border: 2px solid #171717; }
            QPushButton:disabled { background: #ece9e3; color: #aaa49a; border-color: #ece9e3; }
            QPushButton#secondary { background: #fff; color: #292825; border: 1px solid #d9d6d0; border-radius: 22px; padding: 10px 18px; min-height: 20px; }
            QPushButton#secondary:hover { background: #fff7f1; color: #b94f19; border-color: #efcbb4; }
            QLineEdit, QTextEdit { background: #fff; color: #242320; border: 1px solid #d9d6d0; border-radius: 12px; padding: 11px 13px; selection-background-color: #ffd2b8; }
            QLineEdit { font-size: 19px; }
            QTextEdit { font-size: 21px; }
            QLineEdit:focus, QTextEdit:focus { border: 1px solid #ff7a2f; }
            QProgressBar { background: #efede8; border: 0; border-radius: 4px; max-height: 8px; text-align: center; }
            QProgressBar::chunk { background: #ff7a2f; border-radius: 4px; }
            QSlider::groove:horizontal { height: 5px; background: #e4e0d8; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #ff7a2f; border-radius: 2px; }
            QSlider::handle:horizontal { background: white; width: 15px; margin: -6px 0; border: 2px solid #ff7a2f; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #fff0e6; }
            """
        )
        self._update_responsive_layouts()

    @staticmethod
    def _scrollable_tab() -> tuple[QWidget, QWidget]:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        page = QWidget()
        page.setObjectName("tabPage")
        page.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        scroll.setWidget(page)
        tab_layout.addWidget(scroll)
        return tab, page

    def _update_responsive_layouts(self) -> None:
        if not hasattr(self, "sign_columns") or not hasattr(self, "text_sign_columns"):
            return
        width = self.width()
        sign_direction = (
            QBoxLayout.Direction.LeftToRight
            if width >= 1080
            else QBoxLayout.Direction.TopToBottom
        )
        text_sign_direction = (
            QBoxLayout.Direction.LeftToRight
            if width >= 1180
            else QBoxLayout.Direction.TopToBottom
        )
        self.sign_columns.setDirection(sign_direction)
        self.text_sign_columns.setDirection(text_sign_direction)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_layouts()
        if self.sign_reference_pixmap is not None:
            QTimer.singleShot(0, self._scale_sign_reference)

    def _build_sign_tab(self, page: QWidget) -> None:
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(10)
        outer.addWidget(self._page_heading(
            "Sign to text",
            "Show a handshape to see the prediction, then confirm it to build Urdu text.",
        ))
        columns = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        columns.setSpacing(12)
        self.sign_columns = columns

        camera_card, camera_layout = self._card()
        camera_layout.addWidget(self._eyebrow("LIVE INPUT"))
        camera_layout.addWidget(self._section_title("Camera preview"))
        self.camera_preview = QLabel("Camera is off\n\nStart the camera or choose an image")
        self.camera_preview.setObjectName("cameraPreview")
        self.camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setWordWrap(True)
        camera_layout.addWidget(self.camera_preview, stretch=1)
        self.hand_status = QLabel("Hand tracking is off.")
        self.hand_status.setObjectName("mutedText")
        camera_layout.addWidget(self.hand_status)
        camera_controls = QHBoxLayout()
        self.camera_button = QPushButton("Start camera")
        self.camera_button.clicked.connect(self._toggle_camera)
        camera_controls.addWidget(self.camera_button)
        choose_image = QPushButton("Choose image")
        choose_image.setObjectName("secondary")
        choose_image.clicked.connect(self._choose_image)
        camera_controls.addWidget(choose_image)
        camera_layout.addLayout(camera_controls)
        columns.addWidget(camera_card, stretch=2)

        result_card, result_layout = self._card()
        result_layout.addWidget(self._eyebrow("MODEL OUTPUT"))
        result_layout.addWidget(self._section_title("Prediction"))
        self.model_status = QLabel("Checking model files…")
        self.model_status.setObjectName("mutedText")
        self.model_status.setWordWrap(True)
        result_layout.addWidget(self.model_status)
        self.prediction_label = QLabel("Show a hand to see a prediction")
        self.prediction_label.setObjectName("predictionLabel")
        self.prediction_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_label)
        self.confidence_label = QLabel("Confidence: —")
        result_layout.addWidget(self.confidence_label)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        result_layout.addWidget(self.confidence_bar)
        self.character_label = QLabel("—")
        self.character_label.setObjectName("characterLabel")
        self.character_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.character_label)
        self.add_button = QPushButton("Add character to preview")
        self.add_button.setObjectName("secondary")
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self._add_character)
        result_layout.addWidget(self.add_button)
        self.top3_label = QLabel("Top predictions will appear here.")
        self.top3_label.setWordWrap(True)
        result_layout.addWidget(self.top3_label)
        result_layout.addStretch()
        columns.addWidget(result_card, stretch=2)
        outer.addLayout(columns, stretch=3)

        text_card, text_layout = self._card()
        text_layout.addWidget(self._eyebrow("YOUR MESSAGE"))
        text_layout.addWidget(self._section_title("Text preview"))
        self.text_preview = QTextEdit()
        self.text_preview.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.text_preview.setPlaceholderText("Your accepted Urdu characters will appear here.")
        self.text_preview.setMinimumHeight(90)
        text_layout.addWidget(self.text_preview)
        text_controls = QHBoxLayout()
        for label, callback in (
            ("Space", lambda: self.text_preview.insertPlainText(" ")),
            ("Backspace", self._backspace),
            ("Clear", self.text_preview.clear),
            ("Copy text", self._copy_text),
        ):
            button = QPushButton(label)
            button.setObjectName("secondary")
            button.clicked.connect(callback)
            text_controls.addWidget(button)
        text_layout.addLayout(text_controls)
        outer.addWidget(text_card, stretch=1)

    def _build_settings_tab(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._page_heading(
            "Settings",
            "Adjust how predictions are accepted by the text preview.",
        ))
        columns = QHBoxLayout()
        columns.setSpacing(16)

        model_card, model_layout = self._card()
        model_layout.addWidget(self._eyebrow("MODEL"))
        model_layout.addWidget(self._section_title("Model and input"))
        self.checkpoint_status = QLabel(
            f"Checkpoint · {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)}"
        )
        self.checkpoint_status.setObjectName("mutedText")
        self.checkpoint_status.setWordWrap(True)
        model_layout.addWidget(self.checkpoint_status)
        self.mapping_status = QLabel(
            f"Urdu labels · {LABEL_MAPPING.relative_to(PROJECT_ROOT)}"
        )
        self.mapping_status.setObjectName("mutedText")
        self.mapping_status.setWordWrap(True)
        model_layout.addWidget(self.mapping_status)
        model_note = QLabel(
            "The classifier recognizes still handshapes from its 36 trained alphabet classes."
        )
        model_note.setObjectName("mutedText")
        model_note.setWordWrap(True)
        model_layout.addWidget(model_note)
        columns.addWidget(model_card, stretch=3)

        prediction_card, prediction_layout = self._card()
        prediction_layout.addWidget(self._eyebrow("TEXT PREVIEW"))
        prediction_layout.addWidget(self._section_title("Confidence threshold"))
        threshold_note = QLabel(
            "A character is added only when its confidence meets this level. "
            "The live prediction remains visible at any confidence."
        )
        threshold_note.setObjectName("mutedText")
        threshold_note.setWordWrap(True)
        prediction_layout.addWidget(threshold_note)
        self.threshold_label = QLabel("Minimum confidence to add character: 0.50")
        self.threshold_label.setObjectName("mutedText")
        prediction_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(20, 95)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(
            lambda value: self.threshold_label.setText(
                f"Minimum confidence to add character: {value / 100:.2f}"
            )
        )
        prediction_layout.addWidget(self.threshold_slider)
        columns.addWidget(prediction_card, stretch=2)

        # Confirmation method card
        confirm_card, confirm_layout = self._card()
        confirm_layout.addWidget(self._eyebrow("CONFIRMATION"))
        confirm_layout.addWidget(self._section_title("How to accept a predicted letter"))

        confirm_note = QLabel(
            "Choose how a predicted character is added to the text preview while the camera is running.\n"
            "Mouse: click the Add character button. Pinch: pinch with your click hand.\n"
            "Dwell: hold the same sign steady for the set time. Both: either pinch or dwell."
        )
        confirm_note.setObjectName("mutedText")
        confirm_note.setWordWrap(True)
        confirm_layout.addWidget(confirm_note)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.confirm_method_combo = QComboBox()
        self.confirm_method_combo.addItems(["Mouse only", "Pinch gesture", "Dwell hold", "Pinch or dwell"])
        self.confirm_method_combo.currentTextChanged.connect(self._on_confirm_method_changed)
        method_layout.addWidget(self.confirm_method_combo)
        method_layout.addStretch()
        confirm_layout.addLayout(method_layout)

        click_layout = QHBoxLayout()
        click_layout.addWidget(QLabel("Click hand:"))
        self.click_hand_combo = QComboBox()
        self.click_hand_combo.addItems(["Left hand clicks", "Right hand clicks"])
        self.click_hand_combo.currentTextChanged.connect(self._on_click_hand_changed)
        click_layout.addWidget(self.click_hand_combo)
        click_layout.addStretch()
        confirm_layout.addLayout(click_layout)

        dwell_layout = QHBoxLayout()
        dwell_layout.addWidget(QLabel("Dwell time (seconds):"))
        self.dwell_seconds_label = QLabel("1.5")
        self.dwell_seconds_label.setObjectName("mutedText")
        dwell_layout.addWidget(self.dwell_seconds_label)
        self.dwell_seconds_slider = QSlider(Qt.Orientation.Horizontal)
        self.dwell_seconds_slider.setRange(5, 30)
        self.dwell_seconds_slider.setValue(15)
        self.dwell_seconds_slider.valueChanged.connect(self._on_dwell_seconds_changed)
        dwell_layout.addWidget(self.dwell_seconds_slider)
        confirm_layout.addLayout(dwell_layout)

        pinch_layout = QHBoxLayout()
        pinch_layout.addWidget(QLabel("Pinch sensitivity:"))
        self.pinch_threshold_label = QLabel("0.06")
        self.pinch_threshold_label.setObjectName("mutedText")
        pinch_layout.addWidget(self.pinch_threshold_label)
        self.pinch_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.pinch_threshold_slider.setRange(3, 12)
        self.pinch_threshold_slider.setValue(6)
        self.pinch_threshold_slider.valueChanged.connect(self._on_pinch_threshold_changed)
        pinch_layout.addWidget(self.pinch_threshold_slider)
        confirm_layout.addLayout(pinch_layout)

        columns.addWidget(confirm_card, stretch=2)
        layout.addLayout(columns)
        layout.addStretch()

    def _build_library_tab(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._page_heading(
            "Text to signs",
            "Learn the Urdu alphabet by reviewing one sign reference at a time.",
        ))

        columns = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        columns.setSpacing(16)
        self.text_sign_columns = columns

        input_card, input_layout = self._card()
        input_layout.addWidget(self._eyebrow("YOUR TEXT"))
        input_layout.addWidget(self._section_title("Learn signs from Urdu text"))
        message = QLabel(
            "We’ll spell it letter by letter with reference images from the alphabet dataset."
        )
        message.setObjectName("mutedText")
        message.setWordWrap(True)
        input_layout.addWidget(message)
        self.sign_text_input = QLineEdit()
        self.sign_text_input.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.sign_text_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.sign_text_input.setMinimumHeight(52)
        self.sign_text_input.setPlaceholderText("یہاں اردو متن لکھیں")
        self.sign_text_input.returnPressed.connect(self._start_text_sign_sequence)
        input_layout.addWidget(self.sign_text_input)

        # Typing indicator dot - can be clicked with mouse or used with finger touch
        self.typing_dot = QWidget()
        self.typing_dot.setFixedSize(20, 20)
        self.typing_dot.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 100, 100, 200);
                border-radius: 10px;
                border: 2px solid rgba(255, 255, 255, 150);
            }
            QWidget:hover {
                background-color: rgba(255, 150, 150, 255);
            }
            QWidget:pressed {
                background-color: rgba(200, 50, 50, 255);
            }
        """)
        self.typing_dot.setToolTip("Click or tap to indicate typing")
        # Make it focusable and handle both mouse and touch events
        self.typing_dot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.typing_dot.mousePressEvent = self._on_typing_dot_click
        self.typing_dot.installEventFilter(self)

        # Typing dot state label - shows status beside the dot
        self.typing_dot_label = QLabel("\u201c\u201d")
        self.typing_dot_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 200);
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.typing_dot_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Layout for dot + label together
        dot_container = QHBoxLayout()
        dot_container.setContentsMargins(0, 0, 0, 0)
        dot_container.setSpacing(4)
        dot_container.addWidget(self.typing_dot)
        dot_container.addWidget(self.typing_dot_label)
        dot_container_stretch = QHBoxLayout()
        dot_container_stretch.addStretch()

        # Replace single widget with container layout
        # Insert both the dot and its label into the input layout
        # QBoxLayout.addLayout() does not support alignment arg - use stretch to position
        input_layout.addStretch(1)  # push dot container to the right
        input_layout.addLayout(dot_container)
        self.typing_dot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.typing_dot.mousePressEvent = self._on_typing_dot_click
        self.typing_dot.installEventFilter(self)

        self.show_signs_button = QPushButton("Show signs")
        self.show_signs_button.clicked.connect(self._start_text_sign_sequence)
        input_layout.addWidget(self.show_signs_button)
        input_layout.addStretch()
        columns.addWidget(input_card, stretch=2)

        controls_card, controls_layout = self._card()
        controls_layout.addWidget(self._eyebrow("PLAYBACK"))
        controls_layout.addWidget(self._section_title("Review sequence"))
        self.sign_sequence_status = QLabel("Enter Urdu text and select Show signs.")
        self.sign_sequence_status.setObjectName("mutedText")
        controls_layout.addWidget(self.sign_sequence_status)
        self.sign_progress = QProgressBar()
        self.sign_progress.setRange(0, 100)
        self.sign_progress.setValue(0)
        self.sign_progress.setTextVisible(False)
        controls_layout.addWidget(self.sign_progress)

        playback_controls = QHBoxLayout()
        self.sign_previous_button = QPushButton("Previous")
        self.sign_previous_button.setObjectName("secondary")
        self.sign_previous_button.clicked.connect(self._previous_text_sign)
        playback_controls.addWidget(self.sign_previous_button)
        self.sign_play_button = QPushButton("Play")
        self.sign_play_button.clicked.connect(self._toggle_text_sign_playback)
        playback_controls.addWidget(self.sign_play_button)
        self.sign_next_button = QPushButton("Next")
        self.sign_next_button.setObjectName("secondary")
        self.sign_next_button.clicked.connect(self._next_text_sign)
        playback_controls.addWidget(self.sign_next_button)
        controls_layout.addLayout(playback_controls)

        self.sign_speed_label = QLabel("Time per letter: 1.0 seconds")
        self.sign_speed_label.setObjectName("mutedText")
        controls_layout.addWidget(self.sign_speed_label)
        self.sign_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.sign_speed_slider.setRange(3, 30)
        self.sign_speed_slider.setValue(10)
        self.sign_speed_slider.setToolTip(
            "Choose how long each letter's reference image stays visible."
        )
        self.sign_speed_slider.valueChanged.connect(self._update_text_sign_speed)
        controls_layout.addWidget(self.sign_speed_slider)
        speech_controls = QHBoxLayout()
        self.speak_button = QPushButton("Speak text")
        self.speak_button.clicked.connect(self._speak_urdu_text)
        speech_controls.addWidget(self.speak_button)
        self.stop_speech_button = QPushButton("Stop speaking")
        self.stop_speech_button.setObjectName("secondary")
        self.stop_speech_button.clicked.connect(self._stop_urdu_speech)
        speech_controls.addWidget(self.stop_speech_button)
        controls_layout.addLayout(speech_controls)
        self.speech_rate_label = QLabel(
            "Speech rate: 100% (50% slow to 150% fast)"
        )
        self.speech_rate_label.setObjectName("mutedText")
        controls_layout.addWidget(self.speech_rate_label)
        self.speech_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.speech_rate_slider.setRange(
            SPEECH_RATE_MIN_PERCENT,
            SPEECH_RATE_MAX_PERCENT,
        )
        self.speech_rate_slider.setValue(100)
        self.speech_rate_slider.setToolTip(
            "Set Urdu speech speed from 50% (slow) to 150% (fast)."
        )
        self.speech_rate_slider.valueChanged.connect(self._update_speech_rate)
        controls_layout.addWidget(self.speech_rate_slider)
        self.speech_status_label = QLabel("Checking for an Urdu voice…")
        self.speech_status_label.setObjectName("mutedText")
        self.speech_status_label.setWordWrap(True)
        controls_layout.addWidget(self.speech_status_label)
        controls_layout.addStretch()
        self.sign_previous_button.setEnabled(False)
        self.sign_play_button.setEnabled(False)
        self.sign_next_button.setEnabled(False)
        self.speak_button.setEnabled(False)
        self.stop_speech_button.setEnabled(False)
        self.speech_rate_slider.setEnabled(False)
        columns.addWidget(controls_card, stretch=2)

        viewer_card, viewer_layout = self._card()
        viewer_layout.addWidget(self._eyebrow("SIGN PREVIEW"))
        self.sign_current_word_label = QLabel("Your current word will appear here")
        self.sign_current_word_label.setObjectName("mutedText")
        self.sign_current_word_label.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.sign_current_word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewer_layout.addWidget(self.sign_current_word_label)
        self.sign_character_label = QLabel("ا")
        self.sign_character_label.setObjectName("currentLetter")
        self.sign_character_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_character_label.setMinimumHeight(88)
        viewer_layout.addWidget(self.sign_character_label)
        self.sign_class_label = QLabel("Enter text to begin")
        self.sign_class_label.setObjectName("mutedText")
        self.sign_class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewer_layout.addWidget(self.sign_class_label)
        self.sign_image_label = QLabel("Your sign reference will appear here")
        self.sign_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_image_label.setMinimumSize(360, 280)
        self.sign_image_label.setObjectName("cameraPreview")
        self.sign_image_label.setWordWrap(True)
        viewer_layout.addWidget(self.sign_image_label, stretch=1)
        self.sign_reference_note = QLabel(
            "Dataset reference · letter-by-letter fingerspelling"
        )
        self.sign_reference_note.setObjectName("mutedText")
        self.sign_reference_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_reference_note.setWordWrap(True)
        viewer_layout.addWidget(self.sign_reference_note)
        columns.addWidget(viewer_card, stretch=3)

        layout.addLayout(columns, stretch=1)
        self._initialize_urdu_speech()

    def _initialize_urdu_speech(self) -> None:
        try:
            from PySide6.QtCore import QLocale
            from PySide6.QtTextToSpeech import QTextToSpeech
        except ImportError:
            self.speech_status_label.setText(
                "Urdu speech is unavailable because this Qt installation does not include "
                "text-to-speech support."
            )
            return

        try:
            speech = QTextToSpeech(self)
        except RuntimeError as exc:
            self.speech_status_label.setText(f"Could not initialize Urdu speech: {exc}")
            return
        urdu_locales = [
            locale
            for locale in speech.availableLocales()
            if locale.language() == QLocale.Language.Urdu
        ]
        if not urdu_locales:
            self.speech_status_label.setText(
                "No Urdu speech voice is installed on this device. Install an Urdu voice "
                "through your operating system's speech settings, then restart Sign-O."
            )
            return

        urdu_locale = next(
            (locale for locale in urdu_locales if locale.name() == "ur_PK"),
            urdu_locales[0],
        )
        speech.setLocale(urdu_locale)
        urdu_voices = [
            voice
            for voice in speech.availableVoices()
            if voice.language() == QLocale.Language.Urdu
        ]
        if not urdu_voices:
            self.speech_status_label.setText(
                "An Urdu locale is available, but Qt could not find a matching speech "
                "voice. Check your operating system's speech settings."
            )
            return

        speech.setVoice(urdu_voices[0])
        speech.setRate(speech_rate_from_percent(self.speech_rate_slider.value()))
        speech.stateChanged.connect(self._update_speech_status)
        self.speech = speech
        self.speak_button.setEnabled(True)
        self.stop_speech_button.setEnabled(True)
        self.speech_rate_slider.setEnabled(True)
        self.speech_status_label.setText(f"Urdu voice ready: {urdu_voices[0].name()}")

    def _speak_urdu_text(self) -> None:
        if self.speech is None:
            return
        text = self.sign_text_input.text().strip()
        if not text:
            self.speech_status_label.setText("Enter Urdu text before selecting Speak Urdu text.")
            return
        self.speech.stop()
        self.speech.setRate(speech_rate_from_percent(self.speech_rate_slider.value()))
        self.speech.say(text)

    def _stop_urdu_speech(self) -> None:
        if self.speech is not None:
            self.speech.stop()
            self.speech_status_label.setText("Speech stopped.")

    def _update_speech_rate(self, rate_percent: int) -> None:
        self.speech_rate_label.setText(
            f"Speech rate: {rate_percent}% (50% slow to 150% fast)"
        )
        if self.speech is not None:
            self.speech.setRate(speech_rate_from_percent(rate_percent))

    def _update_speech_status(self, state) -> None:
        from PySide6.QtTextToSpeech import QTextToSpeech

        if state == QTextToSpeech.State.Error:
            self.speech_status_label.setText(
                f"Urdu speech failed: {self.speech.errorString()}"
            )
        elif state == QTextToSpeech.State.Speaking:
            self.speech_status_label.setText("Speaking Urdu text…")
        elif state == QTextToSpeech.State.Ready:
            self.speech_status_label.setText("Urdu speech finished or is ready.")

    def _start_text_sign_sequence(self) -> None:
        self.sign_sequence_timer.stop()
        text = self.sign_text_input.text()
        if not text.strip():
            self.sign_sequence_status.setText("Enter Urdu text first.")
            return
        try:
            self.sign_label_mapping = load_label_mapping(LABEL_MAPPING)
            self.sign_example_images = load_example_images(IMAGE_MANIFEST, PROJECT_ROOT)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self.sign_sequence_status.setText(str(exc))
            self._show_error(str(exc))
            return

        self.sign_sequence = split_urdu_text(text, self.sign_label_mapping)
        if not self.sign_sequence:
            self.sign_sequence_status.setText("There are no characters to show.")
            return
        self.sign_sequence_index = 0
        self.sign_play_button.setEnabled(True)
        self.sign_next_button.setEnabled(True)
        self._render_text_sign_step()
        self._toggle_text_sign_playback()

    def _render_text_sign_step(self) -> None:
        if not self.sign_sequence:
            return
        step = self.sign_sequence[self.sign_sequence_index]
        words = self.sign_text_input.text().split()
        word_number = min(
            sum(item.is_space for item in self.sign_sequence[: self.sign_sequence_index]) + 1,
            max(1, len(words)),
        )
        self.sign_sequence_status.setText(
            f"Character {self.sign_sequence_index + 1} of {len(self.sign_sequence)}"
        )
        self.sign_progress.setRange(0, len(self.sign_sequence))
        self.sign_progress.setValue(self.sign_sequence_index + 1)
        current_word = words[word_number - 1] if words else ""
        self.sign_current_word_label.setText(
            f"Word {word_number} of {max(1, len(words))}: {current_word}"
        )
        self.sign_character_label.setText(step.character if not step.is_space else " ")
        self.sign_previous_button.setEnabled(self.sign_sequence_index > 0)
        self.sign_next_button.setEnabled(
            self.sign_sequence_index < len(self.sign_sequence) - 1
        )
        self.sign_image_label.clear()
        self.sign_image_label.setPixmap(QPixmap())
        self.sign_reference_pixmap = None
        self.sign_reference_note.setText(
            "Dataset reference · letter-by-letter fingerspelling"
        )
        if step.is_space:
            self.sign_class_label.setText("Space between words")
            self.sign_image_label.setText("Word break")
            self.sign_reference_note.setText("Word break")
            return
        if step.label is None:
            self.sign_class_label.setText("Unsupported character · outside the 36 dataset classes")
            self.sign_image_label.setText("No matching UAlpha40 sign is available for this character")
            self.sign_reference_note.setText("No verified reference available")
            return

        self.sign_class_label.setText(f"Dataset class: {step.label}")
        image_path = self.sign_example_images.get(step.label)
        if image_path is None:
            if step.label in LANDMARK_CHARACTER_LABELS.values():
                try:
                    pixmap = self.landmark_reference_pixmaps.get(step.label)
                    if pixmap is None:
                        landmark_image = load_landmark_reference(
                            HF_LANDMARK_ROOT,
                            step.label,
                        )
                        rgb = np.asarray(landmark_image.convert("RGB"))
                        height, width, channels = rgb.shape
                        qimage = QImage(
                            rgb.data,
                            width,
                            height,
                            channels * width,
                            QImage.Format.Format_RGB888,
                        ).copy()
                        pixmap = QPixmap.fromImage(qimage)
                        self.landmark_reference_pixmaps[step.label] = pixmap
                except (FileNotFoundError, OSError, ValueError) as exc:
                    self.sign_image_label.setText(
                        f"Could not load landmark reference for {step.label}: {exc}"
                    )
                    self.sign_reference_note.setText("Landmark reference unavailable")
                    return
                self.sign_reference_pixmap = pixmap
                self.sign_reference_note.setText(
                    "HF PSL landmark visualization · static frame, not a photo · "
                    "letter mapping needs PSL-expert confirmation"
                )
                self._scale_sign_reference()
                QTimer.singleShot(0, self._scale_sign_reference)
                return
            self.sign_image_label.setText(
                "Example image not found. Check the processed UAlpha40 image dataset."
            )
            self.sign_reference_note.setText("Dataset image reference unavailable")
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.sign_image_label.setText(f"Could not load dataset image: {image_path}")
            self.sign_reference_note.setText("Dataset image reference unavailable")
            return
        self.sign_reference_pixmap = pixmap
        self.sign_reference_note.setText("UAlpha40 dataset image · static alphabet reference")
        self._scale_sign_reference()
        QTimer.singleShot(0, self._scale_sign_reference)

    def _scale_sign_reference(self) -> None:
        if self.sign_reference_pixmap is None:
            return
        self.sign_image_label.setPixmap(
            self.sign_reference_pixmap.scaled(
                self.sign_image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _toggle_text_sign_playback(self) -> None:
        if not self.sign_sequence:
            return
        if self.sign_sequence_timer.isActive():
            self.sign_sequence_timer.stop()
            self.sign_play_button.setText("Play")
            self.sign_sequence_status.setText(
                f"Paused · character {self.sign_sequence_index + 1} of "
                f"{len(self.sign_sequence)}"
            )
            return
        if self.sign_sequence_index >= len(self.sign_sequence) - 1:
            self.sign_sequence_index = 0
            self._render_text_sign_step()
        self.sign_sequence_timer.start(self.sign_speed_slider.value() * 100)
        self.sign_play_button.setText("Pause")
        self.sign_sequence_status.setText(
            f"Playing · character {self.sign_sequence_index + 1} of "
            f"{len(self.sign_sequence)}"
        )

    def _advance_text_sign(self) -> None:
        if self.sign_sequence_index >= len(self.sign_sequence) - 1:
            self.sign_sequence_timer.stop()
            self.sign_play_button.setText("Play")
            self.sign_sequence_status.setText("Finished. Play again or review a letter.")
            return
        self.sign_sequence_index += 1
        self._render_text_sign_step()
        self.sign_sequence_status.setText(
            f"Playing · character {self.sign_sequence_index + 1} of "
            f"{len(self.sign_sequence)}"
        )

    def _previous_text_sign(self) -> None:
        if not self.sign_sequence or self.sign_sequence_index == 0:
            return
        self.sign_sequence_timer.stop()
        self.sign_play_button.setText("Play")
        self.sign_sequence_index -= 1
        self._render_text_sign_step()

    def _on_typing_dot_click(self, event) -> None:
        """Handle click/tap on the typing indicator dot.
        
        When clicked, this indicates the user has selected/typed a character.
        It can advance the sign sequence or commit the current character to text.
        """
        # Get the current text being typed
        current_text = self.sign_text_input.text().strip()
        
        # If there's text already, adding a dot click could mean:
        # - Commit the current character
        # - Or advance to next prediction
        
        if not current_text:
            # No text yet - this might be the first character selection
            self.keyboard_state.setText("READY - Select a sign")
            self.typing_dot_label.setText("●")  # Filled dot for active state
        else:
            # Text exists - dot click could commit or add space
            # For now, let's add a space or move to next character
            # This behavior can be configured in settings
            self.sign_text_input.setText(current_text + " ")
            self.sign_text_input.setFocus()
            self.typing_dot_label.setText("•")  # Outline dot for idle state
        
        # Visual feedback - change dot color temporarily
        original_style = self.typing_dot.styleSheet()
        self.typing_dot.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 150, 150, 255);
                border-radius: 10px;
                border: 2px solid rgba(255, 255, 255, 150);
            }
            QWidget:hover {
                background-color: rgba(255, 200, 200, 255);
            }
            QWidget:pressed {
                background-color: rgba(200, 50, 50, 255);
            }
        """)
        # Restore original style after short delay
        QTimer.singleShot(150, lambda: self.typing_dot.setStyleSheet(original_style))

    def _next_text_sign(self) -> None:
        if not self.sign_sequence or self.sign_sequence_index >= len(self.sign_sequence) - 1:
            return
        self.sign_sequence_timer.stop()
        self.sign_play_button.setText("Play")
        self.sign_sequence_index += 1
        self._render_text_sign_step()

    def _update_text_sign_speed(self, value: int) -> None:
        seconds = value / 10
        self.sign_speed_label.setText(f"Time per letter: {seconds:.1f} seconds")
        if self.sign_sequence_timer.isActive():
            self.sign_sequence_timer.setInterval(value * 100)

    @staticmethod
    def _eyebrow(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("eyebrow")
        return label

    @staticmethod
    def _page_heading(title_text: str, description: str) -> QWidget:
        heading = QWidget()
        layout = QVBoxLayout(heading)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)
        title = QLabel(title_text)
        title.setObjectName("pageTitle")
        detail = QLabel(description)
        detail.setObjectName("pageDescription")
        detail.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(detail)
        return heading

    @staticmethod
    def _card() -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        return frame, layout

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _load_predictor(self) -> None:
        try:
            self.predictor = MobileNetSignPredictor(DEFAULT_CHECKPOINT, LABEL_MAPPING)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self.model_status.setText(str(exc))
            self.keyboard_state.setText("MODEL NOT READY")
            self.keyboard_state.setObjectName("statusBadgeError")
            return
        self.model_status.setText(
            f"Ready: {len(self.predictor.class_names)} classes on {self.predictor._device}."
        )
        self.keyboard_state.setText("PREVIEW ONLY")

    def _toggle_camera(self) -> None:
        if self.camera is not None:
            self._stop_camera()
            return
        try:
            import cv2
        except ImportError:
            self._show_error("Camera dependency is unavailable. Install requirements-app.txt.")
            return
        try:
            self.hand_tracker = HandTracker()
        except RuntimeError as exc:
            self._show_error(str(exc))
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.hand_tracker.close()
            self.hand_tracker = None
            self._show_error("Could not open the default camera. Check OS camera permission.")
            return
        self.camera_button.setText("Stop camera")
        self.camera_preview.setText("")
        self._last_prediction_time = 0.0
        self.camera_timer.start(80)

    def _stop_camera(self) -> None:
        self.camera_timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.hand_tracker is not None:
            self.hand_tracker.close()
            self.hand_tracker = None
        self.latest_has_hand = False
        self._clear_prediction("Show a hand to see a prediction")
        self.hand_status.setText("Hand tracking is off.")
        self.camera_button.setText("Start camera")
        self.camera_preview.setText("Camera is off")
        self.latest_rgb = None
        self._last_prediction_time = 0.0
        self._dwell_label = None
        self._dwell_start_time = None

    def _update_camera_frame(self) -> None:
        if self.camera is None:
            return
        import cv2

        success, frame = self.camera.read()
        if not success:
            self.camera_preview.setText("Could not read a camera frame.")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.latest_rgb = Image.fromarray(rgb).copy()
        try:
            annotated, hands_info = self.hand_tracker.process(self.latest_rgb)
        except RuntimeError as exc:
            self._stop_camera()
            self._show_error(f"Hand tracking failed: {exc}")
            return
        self.latest_has_hand = len(hands_info) > 0
        self._show_preview(annotated)
        self.hand_status.setText(
            f"Hand detected · {len(hands_info)} hand(s) tracked · green skeleton · red index-finger trail"
            if self.latest_has_hand
            else "No hand detected · show one hand in the camera"
        )
        if self.latest_has_hand and self.predictor is not None:
            now = time.monotonic()
            if now - self._last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
                self._last_prediction_time = now
                self._predict_latest_frame()
            self._handle_gesture_confirmation(hands_info)
        elif not self.latest_has_hand and self.last_prediction is not None:
            self._clear_prediction("Show a hand to see a prediction")
            self._dwell_label = None
            self._dwell_start_time = None

    def _show_preview(self, image: Image.Image) -> None:
        rgb = np.asarray(image.convert("RGB"))
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.camera_preview.setPixmap(
            pixmap.scaled(
                self.camera_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _choose_image(self) -> None:
        if self.camera is not None:
            self._stop_camera()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a sign image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        try:
            with Image.open(path) as image:
                self.latest_rgb = image.convert("RGB").copy()
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not open image: {exc}")
            return
        if self.hand_tracker is None:
            try:
                self.hand_tracker = HandTracker()
            except RuntimeError as exc:
                self._show_error(str(exc))
                return
        try:
            annotated, hands_info = self.hand_tracker.process(self.latest_rgb)
        except RuntimeError as exc:
            self._show_error(f"Hand tracking failed: {exc}")
            return
        self.latest_has_hand = len(hands_info) > 0
        self._show_preview(annotated)
        self.hand_status.setText(
            f"Hand detected · {len(hands_info)} hand(s) tracked · landmarks shown"
            if self.latest_has_hand
            else "No hand detected in this image"
        )
        if self.latest_has_hand:
            self._predict_latest_frame()
        else:
            self._clear_prediction("No hand detected in this image")

    def _predict_latest_frame(self) -> None:
        if self.predictor is None or self.latest_rgb is None or not self.latest_has_hand:
            return
        try:
            prediction = self.predictor.predict_rgb(self.latest_rgb)
        except (RuntimeError, ValueError) as exc:
            self._clear_prediction("Prediction failed")
            if self.camera is not None:
                self._stop_camera()
            self._show_error(f"Prediction failed: {exc}")
            return
        self.last_prediction = prediction
        self.prediction_label.setText(prediction.label)
        self.confidence_label.setText(f"Confidence: {prediction.confidence:.1%}")
        self.confidence_bar.setValue(round(prediction.confidence * 100))
        self.character_label.setText(prediction.character or "Mapping needed")
        threshold = self.threshold_slider.value() / 100
        can_add = prediction.character is not None and prediction.confidence >= threshold
        self.add_button.setEnabled(can_add)
        self.top3_label.setText(
            "Top predictions:\n"
            + "\n".join(f"{label}: {confidence:.1%}" for label, confidence in prediction.top3)
        )

    def _clear_prediction(self, message: str) -> None:
        self.last_prediction = None
        self.prediction_label.setText(message)
        self.confidence_label.setText("Confidence: —")
        self.confidence_bar.setValue(0)
        self.character_label.setText("—")
        self.top3_label.setText("Top predictions will appear here.")
        self.add_button.setEnabled(False)
        self._dwell_label = None
        self._dwell_start_time = None

    # ------------------------------------------------------------------
    # Confirmation gesture handling
    # ------------------------------------------------------------------

    def _click_hand_info(self, hands_info: list) -> object | None:
        """Return the HandInfo for the hand designated as the click hand."""
        for hand in hands_info:
            if self.click_hand == "left" and hand.handedness == "Left":
                return hand
            if self.click_hand == "right" and hand.handedness == "Right":
                return hand
        return None

    def _handle_gesture_confirmation(self, hands_info: list) -> None:
        """Auto-add the current prediction when a confirmation gesture is detected.

        Methods:
          mouse   - never auto-add (user clicks the button)
          pinch   - add when the click hand pinches (thumb + index touch)
          dwell   - add when the same prediction is held steady for dwell_seconds
          both    - either pinch or dwell triggers the add
        """
        if self.confirm_method == "mouse":
            return
        if self.last_prediction is None or self.last_prediction.character is None:
            return

        now = time.monotonic()
        triggered = False

        if self.confirm_method in ("pinch", "both"):
            click_hand = self._click_hand_info(hands_info)
            if click_hand is not None and click_hand.pinch:
                if now - self._last_pinch_time >= PINCH_COOLDOWN_SECONDS:
                    self._last_pinch_time = now
                    triggered = True

        if not triggered and self.confirm_method in ("dwell", "both"):
            current_label = self.last_prediction.label
            if self._dwell_label == current_label and self._dwell_start_time is not None:
                if now - self._dwell_start_time >= self.dwell_seconds:
                    triggered = True
                    self._dwell_start_time = None
                    self._dwell_label = None
            else:
                self._dwell_label = current_label
                self._dwell_start_time = now

        if triggered:
            self._add_character()
            self._dwell_label = None
            self._dwell_start_time = None
            self.character_label.setText(
                f"Added: {self.last_prediction.character}"
            )
            QTimer.singleShot(1200, lambda: self.character_label.setText(
                self.last_prediction.character or "—"
            ))

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        self.confirm_method = data.get("confirm_method", DEFAULT_CONFIRM_METHOD)
        self.dwell_seconds = float(data.get("dwell_seconds", DEFAULT_DWELL_SECONDS))
        self.click_hand = data.get("click_hand", DEFAULT_CLICK_HAND)
        self.pinch_threshold = float(data.get("pinch_threshold", DEFAULT_PINCH_THRESHOLD))
        # Push to UI controls if they exist
        if hasattr(self, "confirm_method_combo"):
            self.confirm_method_combo.setCurrentText(self.confirm_method)
        if hasattr(self, "dwell_seconds_slider"):
            self.dwell_seconds_slider.setValue(int(self.dwell_seconds * 10))
        if hasattr(self, "click_hand_combo"):
            self.click_hand_combo.setCurrentText(self.click_hand)
        if hasattr(self, "pinch_threshold_slider"):
            self.pinch_threshold_slider.setValue(int(self.pinch_threshold * 100))
        if hasattr(self, "hand_tracker") and self.hand_tracker is not None:
            self.hand_tracker.pinch_threshold = self.pinch_threshold

    def _save_settings(self) -> None:
        data = {
            "confirm_method": self.confirm_method,
            "dwell_seconds": self.dwell_seconds,
            "click_hand": self.click_hand,
            "pinch_threshold": self.pinch_threshold,
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Settings UI callbacks
    # ------------------------------------------------------------------

    def _on_confirm_method_changed(self, text: str) -> None:
        mapping = {
            "Mouse only": "mouse",
            "Pinch gesture": "pinch",
            "Dwell hold": "dwell",
            "Pinch or dwell": "both",
        }
        self.confirm_method = mapping.get(text, "mouse")
        self._save_settings()

    def _on_click_hand_changed(self, text: str) -> None:
        if text.startswith("Left"):
            self.click_hand = "left"
        elif text.startswith("Right"):
            self.click_hand = "right"
        self._save_settings()

    def _on_dwell_seconds_changed(self, value: int) -> None:
        self.dwell_seconds = value / 10.0
        self.dwell_seconds_label.setText(f"{self.dwell_seconds:.1f}")
        self._save_settings()

    def _on_pinch_threshold_changed(self, value: int) -> None:
        self.pinch_threshold = value / 100.0
        self.pinch_threshold_label.setText(f"{self.pinch_threshold:.2f}")
        if self.hand_tracker is not None:
            self.hand_tracker.pinch_threshold = self.pinch_threshold
        self._save_settings()

    def _add_character(self) -> None:
        if self.last_prediction is None or self.last_prediction.character is None:
            return
        self.text_preview.insertPlainText(self.last_prediction.character)
        self.text_preview.setFocus()

    def _backspace(self) -> None:
        cursor = self.text_preview.textCursor()
        cursor.deletePreviousChar()
        self.text_preview.setTextCursor(cursor)

    def _copy_text(self) -> None:
        QApplication.clipboard().setText(self.text_preview.toPlainText())

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Sign-O", message)

    def closeEvent(self, event) -> None:
        self._stop_camera()
        self.sign_sequence_timer.stop()
        self._stop_urdu_speech()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = SignOWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""On-device hand landmark visualization for the desktop preview."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def landmarks_to_pixels(
    landmarks: Iterable[object],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    return [
        (
            min(max(round(float(point.x) * width), 0), width - 1),
            min(max(round(float(point.y) * height), 0), height - 1),
        )
        for point in landmarks
    ]


@dataclass
class HandInfo:
    """Information about a single detected hand."""

    handedness: str  # "Left" or "Right"
    landmarks: list[object]  # MediaPipe normalized landmarks
    pixel_points: list[tuple[int, int]]
    pinch: bool
    pinch_distance: float


class HandTracker:
    """Detect up to two hands and draw landmarks, bounding box, and index-finger trails.

    Parameters
    ----------
    trail_length:
        Number of index-fingertip positions kept for the motion trail per hand.
    pinch_threshold:
        Normalized distance below which the thumb and index finger are considered
        to be pinching (a "click" gesture).
    """

    def __init__(self, trail_length: int = 24, pinch_threshold: float = 0.06) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Hand tracking requires MediaPipe. Install requirements-app.txt "
                "in the app's Python 3.12 environment, then restart Python."
            ) from exc

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe installation does not include the Hands tracking API. "
                "Use the pinned mediapipe==0.10.21 app dependency."
            )

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._trails: dict[str, deque[tuple[int, int]]] = {}
        self.pinch_threshold = pinch_threshold

    def _get_trail(self, handedness: str) -> deque[tuple[int, int]]:
        if handedness not in self._trails:
            self._trails[handedness] = deque(maxlen=24)
        return self._trails[handedness]

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, list[HandInfo]]:
        """Process a frame and return the annotated image plus per-hand info."""
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        frame = np.asarray(rgb_image)
        results = self._hands.process(frame)
        annotated = rgb_image.copy()
        draw = ImageDraw.Draw(annotated)

        hands_info: list[HandInfo] = []

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, handedness_list
            ):
                height, width = frame.shape[:2]
                points = landmarks_to_pixels(hand_landmarks.landmark, width, height)
                hand_label = (
                    handedness.classification[0].label
                    if handedness.classification
                    else "Unknown"
                )

                # Pinch detection: distance between thumb tip (4) and index tip (8)
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                pinch_distance = float(
                    ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
                )
                pinch = pinch_distance < self.pinch_threshold

                hands_info.append(
                    HandInfo(
                        handedness=hand_label,
                        landmarks=list(hand_landmarks.landmark),
                        pixel_points=points,
                        pinch=pinch,
                        pinch_distance=pinch_distance,
                    )
                )

                # Draw connections and landmarks for this hand
                connections = self._mp.solutions.hands.HAND_CONNECTIONS
                for start, end in connections:
                    draw.line((points[start], points[end]), fill=(42, 210, 170), width=3)
                for x, y in points:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 255), outline=(25, 68, 120))

                min_x = min(point[0] for point in points)
                max_x = max(point[0] for point in points)
                min_y = min(point[1] for point in points)
                max_y = max(point[1] for point in points)
                draw.rectangle((min_x, min_y, max_x, max_y), outline=(255, 190, 60), width=2)

                # Highlight pinch point when pinching
                if pinch:
                    thumb_px = points[4]
                    index_px = points[8]
                    mid_x = (thumb_px[0] + index_px[0]) // 2
                    mid_y = (thumb_px[1] + index_px[1]) // 2
                    radius = 12
                    draw.ellipse(
                        (mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius),
                        fill=(255, 90, 90),
                        outline=(255, 255, 255),
                        width=3,
                    )
                    draw.line((thumb_px, index_px), fill=(255, 90, 90), width=4)

                # Index-finger trail for this hand
                trail = self._get_trail(hand_label)
                trail.append(points[8])
                if len(trail) > 1:
                    draw.line(list(trail), fill=(255, 90, 90), width=4)
                x, y = points[8]
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 90, 90), outline=(255, 255, 255))

        return annotated, hands_info

    def close(self) -> None:
        self._hands.close()
        self._trails.clear()

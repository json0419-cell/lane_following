import cv2
import numpy as np
from typing import Tuple


DEFAULT_LANE_MASK_MIN_AREA_RATIO = 0.0015
DEFAULT_LANE_MASK_MIN_HEIGHT_RATIO = 0.08
DEFAULT_LANE_MASK_MAX_HORIZONTAL_RATIO = 1.8
DEFAULT_LANE_MASK_MIN_BOTTOM_REACH_RATIO = 0.55
DEFAULT_LANE_MASK_MAX_UPPER_CENTROID_RATIO = 0.38
LANE_MASK_KERNEL = np.ones((3, 3), dtype=np.uint8)

YELLOW_HSV_LOWER = np.array([10, 60, 60], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([40, 255, 255], dtype=np.uint8)
YELLOW_LAB_LOWER = np.array([0, 0, 145], dtype=np.uint8)
YELLOW_LAB_UPPER = np.array([255, 255, 255], dtype=np.uint8)

WHITE_HSV_LOWER = np.array([0, 0, 115], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 85, 255], dtype=np.uint8)
WHITE_LAB_LOWER = np.array([140, 118, 118], dtype=np.uint8)
WHITE_LAB_UPPER = np.array([255, 138, 138], dtype=np.uint8)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={rgb.shape}")
    return rgb


def _postprocess_lane_mask(mask: np.ndarray, *, lane_kind: str) -> np.ndarray:
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={binary.shape}")
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)

    h, w = binary.shape
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, LANE_MASK_KERNEL)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, LANE_MASK_KERNEL)

    min_area = max(6, int(round(h * w * DEFAULT_LANE_MASK_MIN_AREA_RATIO)))
    min_height = max(5, int(round(h * DEFAULT_LANE_MASK_MIN_HEIGHT_RATIO)))
    bottom_keep_threshold = int(round(h * 0.25))
    min_bottom_reach = int(round(h * DEFAULT_LANE_MASK_MIN_BOTTOM_REACH_RATIO))
    max_upper_centroid = float(h * DEFAULT_LANE_MASK_MAX_UPPER_CENTROID_RATIO)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    filtered = np.zeros_like(processed)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        _, cy = centroids[label]
        if area < min_area:
            continue
        if bh < min_height:
            continue
        if bw > max(3, int(round(bh * DEFAULT_LANE_MASK_MAX_HORIZONTAL_RATIO))):
            continue
        if y + bh < bottom_keep_threshold:
            continue

        if lane_kind == "yellow":
            if y + bh < min_bottom_reach and cy < max_upper_centroid and bw > max(4, int(round(bh * 0.65))):
                continue
        else:
            if y + bh < min_bottom_reach and cy < max_upper_centroid:
                continue

        if lane_kind == "yellow" and x > int(round(w * 0.82)):
            continue
        if lane_kind == "white" and x + bw < int(round(w * 0.18)):
            continue

        filtered[labels == label] = 255

    return cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, LANE_MASK_KERNEL)


def _lane_color_masks(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    yellow_hsv_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    yellow_lab_mask = cv2.inRange(lab, YELLOW_LAB_LOWER, YELLOW_LAB_UPPER)
    white_hsv_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white_lab_mask = cv2.inRange(lab, WHITE_LAB_LOWER, WHITE_LAB_UPPER)

    yellow_mask = cv2.bitwise_and(yellow_hsv_mask, yellow_lab_mask)
    white_mask = cv2.bitwise_and(white_hsv_mask, white_lab_mask)
    return yellow_mask, white_mask


def _lane_color_masks_from_bgr(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    yellow_hsv_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    yellow_lab_mask = cv2.inRange(lab, YELLOW_LAB_LOWER, YELLOW_LAB_UPPER)
    white_hsv_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white_lab_mask = cv2.inRange(lab, WHITE_LAB_LOWER, WHITE_LAB_UPPER)

    yellow_mask = cv2.bitwise_and(yellow_hsv_mask, yellow_lab_mask)
    white_mask = cv2.bitwise_and(white_hsv_mask, white_lab_mask)
    return yellow_mask, white_mask


def extract_yellow_lane_mask(image: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(image)
    yellow_mask, _ = _lane_color_masks(rgb)
    return _postprocess_lane_mask(yellow_mask, lane_kind="yellow")


def extract_white_lane_mask(image: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(image)
    _, white_mask = _lane_color_masks(rgb)
    return _postprocess_lane_mask(white_mask, lane_kind="white")


def build_binary_lane_image(image: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(image)
    yellow_mask, white_mask = _lane_color_masks(rgb)
    yellow = _postprocess_lane_mask(yellow_mask, lane_kind="yellow")
    white = _postprocess_lane_mask(white_mask, lane_kind="white")
    merged = np.maximum(yellow, white)
    return np.repeat(merged[..., None], 3, axis=2)


def build_binary_lane_image_from_bgr(image: np.ndarray) -> np.ndarray:
    bgr = np.asarray(image)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HWC BGR image, got shape={bgr.shape}")

    yellow_mask, white_mask = _lane_color_masks_from_bgr(bgr)
    yellow = _postprocess_lane_mask(yellow_mask, lane_kind="yellow")
    white = _postprocess_lane_mask(white_mask, lane_kind="white")
    merged = np.maximum(yellow, white)
    return np.repeat(merged[..., None], 3, axis=2)

"""
Input validation and out-of-distribution detection module.

Provides histogram feature extraction, heuristic chest-X-ray detection,
and a master upload validation pipeline that combines file-type checks,
image quality assessment, and X-ray plausibility scoring.
"""

import io
from typing import Tuple

import numpy as np
from PIL import Image

from utils.dicom_handler import is_dicom, read_dicom
from utils.preprocessing import assess_image_quality


# ---------------------------------------------------------------------------
# Histogram feature extraction
# ---------------------------------------------------------------------------

def compute_image_histogram_features(image: Image.Image) -> dict:
    """Compute histogram-based features for an image.

    Calculates per-channel (R, G, B) mean, standard deviation, skewness,
    and kurtosis, plus an overall grayscale histogram spread metric.

    Args:
        image: Input PIL Image (converted to RGB internally).

    Returns:
        Dictionary with keys:
            - channel_mean (list[float]): Per-channel mean [R, G, B].
            - channel_std (list[float]): Per-channel std deviation.
            - channel_skewness (list[float]): Per-channel skewness.
            - channel_kurtosis (list[float]): Per-channel kurtosis.
            - gray_histogram_spread (float): IQR of the grayscale histogram.
    """
    try:
        rgb = np.array(image.convert("RGB"), dtype=np.float64)

        means, stds, skews, kurts = [], [], [], []
        for ch in range(3):
            channel = rgb[:, :, ch].ravel()
            m = channel.mean()
            s = channel.std()
            means.append(float(m))
            stds.append(float(s))
            # Skewness
            if s > 0:
                skews.append(float(((channel - m) ** 3).mean() / (s ** 3)))
                kurts.append(float(((channel - m) ** 4).mean() / (s ** 4) - 3.0))
            else:
                skews.append(0.0)
                kurts.append(0.0)

        # Grayscale histogram spread (IQR)
        gray = np.array(image.convert("L"), dtype=np.float64).ravel()
        q25, q75 = np.percentile(gray, 25), np.percentile(gray, 75)

        return {
            "channel_mean": means,
            "channel_std": stds,
            "channel_skewness": skews,
            "channel_kurtosis": kurts,
            "gray_histogram_spread": float(q75 - q25),
        }
    except Exception:
        return {
            "channel_mean": [0.0, 0.0, 0.0],
            "channel_std": [0.0, 0.0, 0.0],
            "channel_skewness": [0.0, 0.0, 0.0],
            "channel_kurtosis": [0.0, 0.0, 0.0],
            "gray_histogram_spread": 0.0,
        }


# ---------------------------------------------------------------------------
# Chest X-ray heuristic classifier
# ---------------------------------------------------------------------------

def is_chest_xray(image: Image.Image) -> Tuple[bool, float, str]:
    """Heuristic classifier that estimates whether an image is a chest X-ray.

    Uses four lightweight checks:
        1. **Aspect ratio** — chest X-rays are roughly square to slightly
           portrait; extreme ratios (> 2:1) are rejected.
        2. **Grayscale dominance** — X-rays have very low colour saturation.
        3. **Histogram distribution** — X-rays tend to have a bimodal
           intensity distribution with moderate spread.
        4. **Minimum size** — at least 100 × 100 pixels.

    Args:
        image: Input PIL Image.

    Returns:
        Tuple of (is_xray, confidence, reason) where:
            - is_xray (bool): True if the image is likely a chest X-ray.
            - confidence (float): Score between 0.0 and 1.0.
            - reason (str): Human-readable explanation.
    """
    reasons: list[str] = []
    score = 0.0
    max_score = 4.0  # number of checks

    try:
        width, height = image.size

        # --- Check 1: minimum size ---
        if width >= 100 and height >= 100:
            score += 1.0
        else:
            reasons.append(
                f"Image too small ({width}x{height}); need at least 100x100"
            )

        # --- Check 2: aspect ratio ---
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect <= 2.0:
            score += 1.0
        else:
            reasons.append(
                f"Unusual aspect ratio ({aspect:.1f}:1); "
                "chest X-rays are roughly square"
            )

        # --- Check 3: grayscale dominance (low saturation) ---
        rgb = np.array(image.convert("RGB"), dtype=np.float64)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # Colour saturation: mean absolute difference between channels
        channel_diff = (
            np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()
        ) / 3.0
        if channel_diff < 10.0:
            score += 1.0
        else:
            reasons.append(
                f"High colour saturation ({channel_diff:.1f}); "
                "X-rays are grayscale"
            )

        # --- Check 4: histogram distribution ---
        gray = np.array(image.convert("L"), dtype=np.float64).ravel()
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_norm = hist.astype(np.float64) / hist.sum()

        # X-rays usually have moderate spread (not all-black or all-white)
        spread = float(gray.std())
        if 20.0 < spread < 100.0:
            score += 1.0
        else:
            reasons.append(
                f"Atypical intensity distribution (std={spread:.1f}); "
                "expected 20-100 for chest X-rays"
            )

        confidence = score / max_score
        is_xray = confidence >= 0.5

        if not reasons:
            reason_str = "Image characteristics are consistent with a chest X-ray."
        else:
            reason_str = "; ".join(reasons)

        return is_xray, confidence, reason_str

    except Exception as exc:
        return False, 0.0, f"Validation error: {exc}"


# ---------------------------------------------------------------------------
# Master upload validator
# ---------------------------------------------------------------------------

def validate_upload(file_bytes: bytes, filename: str) -> dict:
    """Validate an uploaded file for the pneumonia detection pipeline.

    Performs file-type detection (DICOM vs. standard image), image quality
    assessment, and X-ray plausibility checking in one call.

    Args:
        file_bytes: Raw uploaded file bytes.
        filename: Original filename (used for format detection).

    Returns:
        Dictionary with keys:
            - is_valid (bool): Whether the upload is usable.
            - is_dicom (bool): Whether the file is DICOM format.
            - image (PIL.Image or None): Decoded image on success.
            - metadata (dict or None): DICOM metadata if applicable.
            - quality (dict): Image quality assessment results.
            - xray_check (tuple): Result of is_chest_xray().
            - warnings (list[str]): Non-blocking issues.
            - errors (list[str]): Blocking issues.
    """
    result: dict = {
        "is_valid": False,
        "is_dicom": False,
        "image": None,
        "metadata": None,
        "quality": {
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "is_acceptable": False,
            "issues": [],
        },
        "xray_check": (False, 0.0, "Not checked"),
        "warnings": [],
        "errors": [],
    }

    if not file_bytes:
        result["errors"].append("Empty file uploaded.")
        return result

    # ------ DICOM detection & decoding ------
    dicom_detected = is_dicom(file_bytes)
    result["is_dicom"] = dicom_detected

    pil_image = None
    if dicom_detected:
        try:
            pil_image, metadata = read_dicom(file_bytes)
            result["metadata"] = metadata
        except ImportError as exc:
            result["errors"].append(str(exc))
            return result
        except Exception as exc:
            result["errors"].append(f"Failed to read DICOM file: {exc}")
            return result
    else:
        # Try standard image formats
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            pil_image.load()  # force decode
            pil_image = pil_image.convert("RGB")
        except Exception:
            result["errors"].append(
                "Unsupported image format. Please upload a JPEG, PNG, or "
                "DICOM (.dcm) file."
            )
            return result

    result["image"] = pil_image

    # ------ Image quality assessment ------
    quality = assess_image_quality(pil_image)
    result["quality"] = quality
    if not quality["is_acceptable"]:
        for issue in quality["issues"]:
            result["warnings"].append(f"Quality issue: {issue}")

    # ------ Chest X-ray plausibility ------
    xray_result = is_chest_xray(pil_image)
    result["xray_check"] = xray_result
    is_xray, confidence, reason = xray_result

    if not is_xray:
        result["warnings"].append(
            f"This image may not be a chest X-ray (confidence: "
            f"{confidence:.0%}). Reason: {reason}"
        )

    # ------ Final decision ------
    # Valid as long as we have a decodable image; quality / xray issues are
    # warnings, not hard errors.
    result["is_valid"] = True

    return result

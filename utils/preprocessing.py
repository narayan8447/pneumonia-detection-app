"""
CLAHE-based X-ray image enhancement and preprocessing module.

Provides contrast-limited adaptive histogram equalization (CLAHE) for
improving X-ray visibility, image quality assessment, and a full
preprocessing pipeline that prepares images for the CNN model.
"""

import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
from PIL import Image, ImageOps


def apply_clahe(image: Image.Image, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Converts the input to grayscale, applies CLAHE via OpenCV (or PIL histogram
    equalization fallback if OpenCV is not installed), and returns the enhanced
    image converted back to RGB.

    Args:
        image: Input PIL Image (any mode).
        clip_limit: Threshold for contrast limiting. Higher values give more
            contrast but may amplify noise.  Defaults to 2.0.
        tile_grid: Tuple (rows, cols) defining the number of tiles for local
            histogram equalization.  Defaults to (8, 8).

    Returns:
        Enhanced PIL Image in RGB mode.
    """
    try:
        gray = image.convert("L")
        if _HAS_CV2:
            img_array = np.array(gray, dtype=np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid,
            )
            enhanced = clahe.apply(img_array)
            return Image.fromarray(enhanced).convert("RGB")
        else:
            # Fallback: PIL adaptive equalization
            return ImageOps.equalize(gray).convert("RGB")
    except Exception:
        # Fallback: return original image converted to RGB
        return image.convert("RGB")


def assess_image_quality(image: Image.Image) -> dict:
    """Assess the quality of an X-ray image.

    Computes blur, brightness, and contrast metrics and flags common issues.

    Args:
        image: Input PIL Image.

    Returns:
        Dictionary with keys:
            - blur_score (float): Laplacian variance — higher is sharper.
            - brightness (float): Mean pixel intensity (0-255).
            - contrast (float): Standard deviation of pixel intensities.
            - is_acceptable (bool): True when no blocking issues are found.
            - issues (list[str]): Human-readable list of detected problems.
    """
    issues: list[str] = []

    try:
        gray = np.array(image.convert("L"), dtype=np.float64)

        # Blur detection via Laplacian variance
        if _HAS_CV2:
            laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
            blur_score = float(laplacian.var())
        else:
            # Pure NumPy 3x3 Laplacian filter fallback
            if gray.shape[0] >= 3 and gray.shape[1] >= 3:
                lap = (
                    gray[:-2, 1:-1]
                    + gray[2:, 1:-1]
                    + gray[1:-1, :-2]
                    + gray[1:-1, 2:]
                    - 4.0 * gray[1:-1, 1:-1]
                )
                blur_score = float(lap.var())
            else:
                blur_score = 100.0

        # Brightness & contrast
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # Thresholds (tuned for chest X-ray images)
        if blur_score < 40.0:
            issues.append("Too blurry")
        if brightness < 25.0:
            issues.append("Underexposed")
        elif brightness > 230.0:
            issues.append("Overexposed")
        if contrast < 15.0:
            issues.append("Low contrast")

        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "is_acceptable": len(issues) == 0,
            "issues": issues,
        }
    except Exception:
        return {
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "is_acceptable": False,
            "issues": ["Unable to assess image quality"],
        }


def preprocess_xray(image: Image.Image,
                    apply_enhancement: bool = True) -> Image.Image:
    """Full preprocessing pipeline for chest X-ray images.

    Steps:
        1. Apply EXIF orientation transpose.
        2. Convert to RGB.
        3. Optionally apply CLAHE enhancement.
        4. Resize and center-crop to 224×224.

    Args:
        image: Input PIL Image.
        apply_enhancement: Whether to apply CLAHE.  Defaults to True.

    Returns:
        Preprocessed 224×224 RGB PIL Image ready for model inference.
    """
    target_size = 224

    try:
        # 1. Correct EXIF orientation
        image = ImageOps.exif_transpose(image)

        # 2. Ensure RGB
        image = image.convert("RGB")

        # 3. Optional CLAHE
        if apply_enhancement:
            image = apply_clahe(image)

        # 4. Resize (shortest side -> target_size) then center-crop
        width, height = image.size
        if width < height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))

        return image
    except Exception:
        # Fallback: brute-force resize
        return image.convert("RGB").resize(
            (target_size, target_size), Image.LANCZOS
        )

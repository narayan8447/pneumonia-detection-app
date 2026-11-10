"""
DICOM file handling module.

Provides utilities for detecting DICOM files, reading pixel data and
metadata from .dcm files via pydicom, and formatting metadata for display.
"""

import io
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


def is_dicom(file_bytes: bytes) -> bool:
    """Check whether the given bytes represent a DICOM file.

    DICOM files contain the magic bytes ``DICM`` at byte offset 128.

    Args:
        file_bytes: Raw file content.

    Returns:
        True if the DICOM preamble is detected.
    """
    try:
        if len(file_bytes) < 132:
            return False
        return file_bytes[128:132] == b"DICM"
    except Exception:
        return False


def read_dicom(file_bytes: bytes) -> Tuple[Image.Image, dict]:
    """Read a DICOM file from raw bytes.

    Extracts the pixel array, applies VOI LUT when available, normalises
    to 8-bit, and collects common DICOM metadata tags.

    Args:
        file_bytes: Raw DICOM file content.

    Returns:
        A tuple of (PIL RGB Image, metadata_dict).  The metadata dict
        contains the keys: patient_name, patient_id, patient_age,
        patient_sex, study_date, modality, body_part, view_position,
        institution, manufacturer, image_size.

    Raises:
        ImportError: If pydicom is not installed.
        ValueError: If pixel data cannot be extracted.
    """
    if not _HAS_PYDICOM:
        raise ImportError(
            "pydicom is required to read DICOM files. "
            "Install it with: pip install pydicom"
        )

    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))

        # Extract pixel array
        pixel_array = ds.pixel_array.astype(np.float64)

        # Apply VOI LUT if available
        try:
            pixel_array = apply_voi_lut(pixel_array, ds).astype(np.float64)
        except Exception:
            pass  # VOI LUT not available; continue with raw pixels

        # Handle PhotometricInterpretation (invert if MONOCHROME1)
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

        # Normalise to 8-bit
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max - p_min > 0:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min) * 255.0)
        pixel_array = pixel_array.astype(np.uint8)

        # Convert to PIL RGB
        pil_image = Image.fromarray(pixel_array).convert("RGB")

        # Collect metadata
        metadata = {
            "patient_name": str(getattr(ds, "PatientName", "N/A")),
            "patient_id": str(getattr(ds, "PatientID", "N/A")),
            "patient_age": str(getattr(ds, "PatientAge", "N/A")),
            "patient_sex": str(getattr(ds, "PatientSex", "N/A")),
            "study_date": str(getattr(ds, "StudyDate", "N/A")),
            "modality": str(getattr(ds, "Modality", "N/A")),
            "body_part": str(getattr(ds, "BodyPartExamined", "N/A")),
            "view_position": str(getattr(ds, "ViewPosition", "N/A")),
            "institution": str(getattr(ds, "InstitutionName", "N/A")),
            "manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
            "image_size": f"{pixel_array.shape[1]}x{pixel_array.shape[0]}",
        }

        return pil_image, metadata

    except ImportError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read DICOM data: {exc}") from exc


def format_dicom_metadata(meta: dict) -> str:
    """Format a DICOM metadata dictionary into a human-readable string.

    Args:
        meta: Metadata dictionary as returned by :func:`read_dicom`.

    Returns:
        A nicely formatted multi-line string suitable for display.
    """
    label_map = {
        "patient_name": "Patient Name",
        "patient_id": "Patient ID",
        "patient_age": "Patient Age",
        "patient_sex": "Patient Sex",
        "study_date": "Study Date",
        "modality": "Modality",
        "body_part": "Body Part",
        "view_position": "View Position",
        "institution": "Institution",
        "manufacturer": "Manufacturer",
        "image_size": "Image Size",
    }

    lines = ["═══ DICOM Metadata ═══"]
    for key, label in label_map.items():
        value = meta.get(key, "N/A")
        lines.append(f"  {label:<18}: {value}")
    lines.append("══════════════════════")

    return "\n".join(lines)

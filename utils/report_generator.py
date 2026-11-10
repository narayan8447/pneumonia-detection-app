"""
PDF diagnostic report generator.

Produces a professional AI-assisted chest X-ray analysis report using
fpdf2, including patient information, side-by-side images, prediction
results, severity breakdown, clinical notes, and a mandatory disclaimer.
"""

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False


def generate_pdf_report(
    original_image: Image.Image,
    gradcam_overlay: Image.Image,
    prediction: str,
    confidence: float,
    severity_info: dict,
    metadata: dict | None = None,
    patient_info: dict | None = None,
) -> bytes:
    """Generate a professional PDF diagnostic report.

    The report contains:
        - Title and generation timestamp.
        - Patient information (from DICOM metadata or manual input).
        - Side-by-side images: original X-ray and Grad-CAM overlay.
        - Prediction results with classification, confidence, and severity.
        - Severity breakdown table by region.
        - Clinical notes.
        - Mandatory AI disclaimer.

    Args:
        original_image: Original chest X-ray as a PIL Image.
        gradcam_overlay: Grad-CAM overlay as a PIL Image.
        prediction: Classification result string (e.g. ``'PNEUMONIA'``).
        confidence: Prediction confidence as a float 0-1.
        severity_info: Severity dictionary from
            :func:`utils.severity.estimate_severity`.
        metadata: Optional DICOM metadata dictionary.
        patient_info: Optional manually entered patient information
            dictionary with keys like ``name``, ``age``, ``sex``, ``id``.

    Returns:
        The generated PDF file as raw bytes.

    Raises:
        ImportError: If fpdf2 is not installed.
    """
    if not _HAS_FPDF:
        raise ImportError(
            "fpdf2 is required to generate PDF reports. "
            "Install it with: pip install fpdf2"
        )

    now = datetime.now(tz=timezone.utc)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Header ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI-Assisted Chest X-Ray Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {timestamp_str}", ln=True, align="C")
    pdf.ln(6)
    _draw_line(pdf)

    # ── Patient Information ──────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 10)

    patient_fields = _collect_patient_fields(metadata, patient_info)
    for label, value in patient_fields:
        pdf.cell(55, 6, f"{label}:", ln=False)
        pdf.cell(0, 6, str(value), ln=True)
    pdf.ln(4)
    _draw_line(pdf)

    # ── Images (side by side) ────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Imaging", ln=True)

    tmp_files: list[str] = []
    try:
        orig_path = _save_temp_image(original_image, "orig")
        cam_path = _save_temp_image(gradcam_overlay, "cam")
        tmp_files.extend([orig_path, cam_path])

        img_w = 85  # mm width per image
        y_before = pdf.get_y()

        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(img_w, 5, "Original X-Ray", align="C")
        pdf.cell(10)  # spacer
        pdf.cell(img_w, 5, "Grad-CAM Overlay", align="C", ln=True)

        pdf.image(orig_path, x=pdf.get_x(), y=pdf.get_y(), w=img_w)
        pdf.image(cam_path, x=pdf.get_x() + img_w + 10, y=pdf.get_y(), w=img_w)
        pdf.ln(img_w + 5)  # approximate height
    except Exception:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, "[Images could not be embedded]", ln=True)
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    _draw_line(pdf)

    # ── Prediction Results ───────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Prediction Results", ln=True)
    pdf.set_font("Helvetica", "", 11)

    conf_pct = confidence * 100.0 if confidence <= 1.0 else confidence
    severity_level = severity_info.get("severity_level", "N/A")
    affected_pct = severity_info.get("affected_percentage", 0.0)

    pdf.cell(55, 7, "Classification:", ln=False)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, prediction, ln=True)
    pdf.set_font("Helvetica", "", 11)

    pdf.cell(55, 7, "Confidence:", ln=False)
    pdf.cell(0, 7, f"{conf_pct:.1f}%", ln=True)

    pdf.cell(55, 7, "Severity Level:", ln=False)
    pdf.cell(0, 7, severity_level, ln=True)

    pdf.cell(55, 7, "Affected Area:", ln=False)
    pdf.cell(0, 7, f"{affected_pct:.1f}%", ln=True)
    pdf.ln(4)
    _draw_line(pdf)

    # ── Severity Breakdown ───────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Severity Breakdown by Region", ln=True)

    regions = severity_info.get("affected_regions", {})
    if regions:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, "Region", border=1)
        pdf.cell(50, 7, "Affected (%)", border=1, ln=True)
        pdf.set_font("Helvetica", "", 10)
        for region_name, region_pct in regions.items():
            pdf.cell(70, 7, region_name, border=1)
            pdf.cell(50, 7, f"{region_pct:.1f}%", border=1, ln=True)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, "No region data available.", ln=True)

    pdf.ln(4)
    _draw_line(pdf)

    # ── Clinical Notes ───────────────────────────────────────────────
    clinical_note = severity_info.get("clinical_note", "")
    if clinical_note:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Clinical Notes", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, clinical_note)
        pdf.ln(4)
        _draw_line(pdf)

    # ── Disclaimer ───────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "DISCLAIMER", ln=True)
    pdf.set_font("Helvetica", "", 9)
    disclaimer = (
        "DISCLAIMER: This is an AI-assisted preliminary screening tool. "
        "Results must be reviewed and confirmed by a qualified radiologist "
        "or physician. This tool is not FDA-approved and should not be used "
        "as the sole basis for clinical decisions."
    )
    pdf.multi_cell(0, 5, disclaimer)
    pdf.ln(4)

    # ── Footer ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0, 5,
        f"Report generated on {timestamp_str} by AI-Assisted Chest X-Ray "
        f"Analysis System.",
        ln=True,
        align="C",
    )

    # Output PDF safely across fpdf2 and legacy fpdf
    try:
        out = pdf.output(dest="S")
        if isinstance(out, str):
            return out.encode("latin1")
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
    except TypeError:
        # Some fpdf2 versions do not accept dest keyword
        out = pdf.output()
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        if isinstance(out, str):
            return out.encode("latin1")
    return bytes(out)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _draw_line(pdf: "FPDF") -> None:
    """Draw a horizontal separator line across the page.

    Args:
        pdf: Active FPDF instance.
    """
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)


def _save_temp_image(image: Image.Image, prefix: str) -> str:
    """Save a PIL Image to a temporary PNG file and return its path.

    Args:
        image: PIL Image to save.
        prefix: Filename prefix for the temp file.

    Returns:
        Absolute path to the created temporary PNG file.
    """
    fd, path = tempfile.mkstemp(suffix=".png", prefix=f"report_{prefix}_")
    os.close(fd)
    image.convert("RGB").save(path, format="PNG")
    return path


def _collect_patient_fields(
    metadata: dict | None,
    patient_info: dict | None,
) -> list[tuple[str, str]]:
    """Merge DICOM metadata and manual patient info into display fields.

    Manual ``patient_info`` values take precedence over DICOM ``metadata``.

    Args:
        metadata: DICOM metadata dict (may be None).
        patient_info: Manually entered patient info dict (may be None).

    Returns:
        List of (label, value) tuples for display.
    """
    fields: list[tuple[str, str]] = []

    # Helper to pull a value from patient_info first, then metadata
    def _get(pi_key: str, meta_key: str, label: str) -> None:
        val = "N/A"
        if patient_info and patient_info.get(pi_key):
            val = str(patient_info[pi_key])
        elif metadata and metadata.get(meta_key) and metadata[meta_key] != "N/A":
            val = str(metadata[meta_key])
        fields.append((label, val))

    _get("name", "patient_name", "Patient Name")
    _get("id", "patient_id", "Patient ID")
    _get("age", "patient_age", "Age")
    _get("sex", "patient_sex", "Sex")

    # Extra DICOM-only fields
    if metadata:
        for meta_key, label in [
            ("study_date", "Study Date"),
            ("modality", "Modality"),
            ("body_part", "Body Part"),
            ("institution", "Institution"),
        ]:
            val = metadata.get(meta_key, "N/A")
            if val and val != "N/A":
                fields.append((label, str(val)))

    return fields

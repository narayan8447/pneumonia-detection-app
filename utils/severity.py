"""
Severity estimation module for Grad-CAM heatmaps.

Analyses a Grad-CAM activation map to estimate the spatial extent and
severity of pneumonia-like findings, and generates a clinical note
summarising the results.
"""

import numpy as np


def estimate_severity(cam: np.ndarray, threshold: float = 0.4) -> dict:
    """Estimate pneumonia severity from a Grad-CAM heatmap.

    Divides the heatmap into five anatomical regions (Upper Left,
    Upper Right, Lower Left, Lower Right, Central) and computes the
    percentage of each region that exceeds the activation threshold.

    Severity levels:
        - **Mild**: < 20 % of the total area affected.
        - **Moderate**: 20 – 50 % affected.
        - **Severe**: > 50 % affected.

    Args:
        cam: 2-D NumPy array with values in [0, 1] representing the
            Grad-CAM heatmap.
        threshold: Activation threshold above which a pixel is considered
            "affected".  Defaults to 0.4.

    Returns:
        Dictionary with keys:
            - affected_percentage (float): Overall affected area, 0-100.
            - severity_level (str): ``'Mild'``, ``'Moderate'``, or
              ``'Severe'``.
            - severity_score (float): Normalised severity score, 0-1.
            - affected_regions (dict[str, float]): Per-region affected
              percentage.
            - clinical_note (str): Human-readable summary of findings.
    """
    try:
        cam = np.asarray(cam, dtype=np.float64)

        # Ensure 2-D
        if cam.ndim != 2:
            return _default_result("Invalid heatmap dimensions.")

        h, w = cam.shape
        if h == 0 or w == 0:
            return _default_result("Empty heatmap.")

        # Clip values to [0, 1]
        cam = np.clip(cam, 0.0, 1.0)

        # Binary mask of affected pixels
        affected_mask = cam >= threshold

        # --- Overall percentage ---
        total_pixels = h * w
        affected_pixels = int(affected_mask.sum())
        affected_percentage = (affected_pixels / total_pixels) * 100.0

        # --- Per-region analysis ---
        mid_h, mid_w = h // 2, w // 2
        # Central region: middle 50 % of each axis
        c_top = h // 4
        c_bot = h - h // 4
        c_left = w // 4
        c_right = w - w // 4

        regions = {
            "Upper Left": affected_mask[:mid_h, :mid_w],
            "Upper Right": affected_mask[:mid_h, mid_w:],
            "Lower Left": affected_mask[mid_h:, :mid_w],
            "Lower Right": affected_mask[mid_h:, mid_w:],
            "Central": affected_mask[c_top:c_bot, c_left:c_right],
        }

        affected_regions: dict[str, float] = {}
        for name, region in regions.items():
            region_total = region.size
            if region_total > 0:
                affected_regions[name] = float(
                    (region.sum() / region_total) * 100.0
                )
            else:
                affected_regions[name] = 0.0

        # --- Severity classification ---
        if affected_percentage < 20.0:
            severity_level = "Mild"
        elif affected_percentage <= 50.0:
            severity_level = "Moderate"
        else:
            severity_level = "Severe"

        severity_score = min(affected_percentage / 100.0, 1.0)

        # --- Clinical note ---
        clinical_note = _build_clinical_note(
            affected_percentage, severity_level, affected_regions
        )

        return {
            "affected_percentage": round(affected_percentage, 2),
            "severity_level": severity_level,
            "severity_score": round(severity_score, 4),
            "affected_regions": {
                k: round(v, 2) for k, v in affected_regions.items()
            },
            "clinical_note": clinical_note,
        }

    except Exception as exc:
        return _default_result(f"Severity estimation error: {exc}")


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _build_clinical_note(
    pct: float, level: str, regions: dict[str, float]
) -> str:
    """Compose a human-readable clinical note from severity data.

    Args:
        pct: Overall affected percentage.
        level: Severity level string.
        regions: Per-region percentages.

    Returns:
        Multi-line clinical note string.
    """
    lines: list[str] = []
    lines.append(
        f"AI-estimated severity: {level} "
        f"({pct:.1f}% of lung field shows increased opacity)."
    )

    # Identify the most-affected regions (> 10 %)
    hot_regions = [name for name, val in regions.items() if val > 10.0]
    if hot_regions:
        lines.append(
            f"Predominant involvement in: {', '.join(hot_regions)}."
        )
    else:
        lines.append("No region shows significant focal involvement.")

    # Severity-specific guidance
    if level == "Mild":
        lines.append(
            "Findings suggest minor or early-stage opacification. "
            "Clinical correlation recommended."
        )
    elif level == "Moderate":
        lines.append(
            "Findings suggest moderate opacification affecting a "
            "substantial portion of the lung field. Further evaluation "
            "with lateral view or CT may be warranted."
        )
    else:
        lines.append(
            "Findings suggest extensive opacification. Urgent clinical "
            "evaluation and possible advanced imaging are recommended."
        )

    return " ".join(lines)


def _default_result(note: str) -> dict:
    """Return a safe default severity result with the given note.

    Args:
        note: Explanatory message to include as the clinical note.

    Returns:
        Severity result dictionary with zeroed-out values.
    """
    return {
        "affected_percentage": 0.0,
        "severity_level": "Mild",
        "severity_score": 0.0,
        "affected_regions": {
            "Upper Left": 0.0,
            "Upper Right": 0.0,
            "Lower Left": 0.0,
            "Lower Right": 0.0,
            "Central": 0.0,
        },
        "clinical_note": note,
    }

# 🫁 Pneumonia X-Ray Screening — Intelligent Chest X-Ray Analysis

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI-powered chest X-ray screening tool that classifies images as **Normal** or **Pneumonia** with explainable Grad-CAM visualizations, severity assessment, and professional PDF report generation.
---

## ✨ Features

- 🔬 **Deep Learning Classification** — Normal vs Pneumonia detection using a custom lightweight CNN
- 🔥 **Grad-CAM Explainability** — Visual heatmaps highlighting the regions driving the model's decision
- 📊 **Severity Assessment** — Mild / Moderate / Severe grading with regional lung analysis
- 🏥 **DICOM Support** — Native `.dcm` file handling with full metadata extraction
- ✨ **CLAHE Enhancement** — Adaptive contrast-limited histogram equalization for improved visualization
- 🛡️ **Smart Input Validation** — Out-of-distribution detection and image quality checks
- 📄 **PDF Report Generation** — Professional one-click diagnostic reports
- 📈 **Analytics Dashboard** — Session statistics and prediction trend tracking
- 🎨 **Modern UI** — Glass-morphism design with a dark medical-grade theme

---

## 🖥️ Demo

> **Live Demo:** *(https://pneumoscann.streamlit.app/)*

<!-- Add a screenshot here -->
<!-- ![Pneumonia X-Ray Screening Screenshot](docs/screenshot.png) -->

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning inference & Grad-CAM computation |
| **Streamlit** | Interactive web application framework |
| **OpenCV** | Image processing & CLAHE enhancement |
| **pydicom** | DICOM medical image file parsing |
| **fpdf2** | PDF report generation |
| **NumPy / Pandas** | Numerical computing & analytics |
| **Pillow** | Image I/O and manipulation |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/narayan8447/pneumonia-xrayscreening.git
cd pneumonia-xrayscreening

# Install dependencies (uses CPU-only PyTorch by default)
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
pneumonia-xrayscreening/
├── streamlit_app.py          # Main Streamlit application
├── arch.py                   # CNN model architecture definition
├── xray_model.pth            # Pre-trained model weights
├── requirements.txt          # Python dependencies (CPU PyTorch)
├── packages.txt              # System packages for Streamlit Cloud
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # Streamlit theme & server configuration
└── utils/
    ├── __init__.py            # Package initializer
    ├── preprocessing.py       # CLAHE enhancement & image preprocessing
    ├── dicom_handler.py       # DICOM file reading & metadata extraction
    ├── input_validator.py     # Input validation & OOD detection
    ├── severity.py            # Severity estimation from Grad-CAM
    └── report_generator.py    # PDF diagnostic report generation
```

---

## 🧠 Model Architecture

Pneumonia X-Ray Screening uses a **custom lightweight CNN** optimized for chest X-ray classification:

- **Architecture:** 9-block convolutional neural network with batch normalization and dropout
- **Input:** 224 × 224 RGB images
- **Output:** 2 classes — `Normal` and `Pneumonia`
- **Parameters:** ~15K (designed for fast CPU inference)
- **Training:** Trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset

The compact architecture enables real-time inference even on CPU-only deployments like Streamlit Cloud.

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A[📤 Image Upload] --> B[🛡️ Validation]
    B --> C[✨ CLAHE Preprocessing]
    C --> D[🧠 CNN Inference]
    D --> E[🔥 Grad-CAM Generation]
    E --> F[📊 Severity Estimation]
    F --> G[📄 Report Generation]
```

1. **Image Upload & Validation** — Upload a chest X-ray (JPEG, PNG, or DICOM). The system validates the image format, quality, and checks whether it looks like a genuine chest X-ray.
2. **CLAHE Preprocessing** *(optional)* — Adaptive contrast enhancement improves visibility of subtle features in the X-ray.
3. **CNN Inference** — The pre-trained model classifies the image as Normal or Pneumonia with a confidence score.
4. **Grad-CAM Generation** — Gradient-weighted Class Activation Mapping highlights the lung regions influencing the prediction.
5. **Severity Estimation** — Analyzes the Grad-CAM heatmap to estimate affected area percentage and severity level (Mild / Moderate / Severe).
6. **Report Generation** — Generate a downloadable PDF report containing all findings, images, and clinical notes.

---

## ☁️ Deployment

### Streamlit Cloud

This project is pre-configured for **Streamlit Cloud** deployment:

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repository.
3. Set **Main file path** to `streamlit_app.py`.
4. Deploy!

> [!NOTE]
> The `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cpu` to install CPU-only PyTorch, keeping the deployment size small. The `packages.txt` file provides the system-level dependencies (libgl, libglib) required by OpenCV on Linux.

### Local Deployment with GPU

If you have a CUDA-capable GPU and want GPU acceleration, install PyTorch with CUDA support separately:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

> [!CAUTION]
> **This is an AI-assisted preliminary screening tool.** Results generated by Pneumonia X-Ray Screening must be reviewed and confirmed by a qualified radiologist or physician. This tool is **not FDA-approved** and should **not** be used as the sole basis for clinical decisions. Always consult a healthcare professional for medical diagnosis and treatment.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Narayan** — [GitHub](https://github.com/narayan8447)

---

<p align="center">
  Made with ❤️ for accessible healthcare AI
</p>

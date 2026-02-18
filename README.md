---
title: AutoSeg Terrain Safety
emoji: 🛡️
colorFrom: green
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Terrain Safety Analysis with SegFormer (AutoSeg)

**Real-time Semantic Segmentation for Safe Navigation Assistance**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-SegFormer-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Project Overview

**AutoSeg** is a lightweight, real-time semantic segmentation prototype designed to analyze unstructured terrain images for autonomous navigation safety.

Using **SegFormer** transformer models, it segments scenes into 150 categories (ADE20K) and maps them to a customizable **Safety Score**. The system visualizes safe paths (Green) vs. hazards (Red) via a Heads-Up Display (HUD) overlay, providing actionable insights for mobile robots or assistive navigation devices.

## ✨ Key Features

- **🚀 Efficient Inference**: Supports SegFormer B0 (Fast) and B2 (Balanced) for real-time CPU/GPU performance.
- **👁️ Depth Perception**: Monocular depth estimation using **Depth Anything V2** to understand terrain geometry.
- **📍 Smart Pathfinding**: A\* navigation with "Soft Cost" logic to find optimal ground paths while avoiding obstacles.
- **🧊 3D Visualization**: Interactive "Sci-Fi" mesh visualization of the terrain and computed path.
- **🔧 Safety Logic Engine**: Fully configurable JSON mapping to define what counts as "Safe" vs. "Hazard" (e.g., treat 'water' as hazard).
- **📊 Observability**: Built-in [Weights & Biases](https://wandb.ai) logging for experiment tracking.
- **☁️ Deployment Ready**: Optimized for Hugging Face Spaces (Gradio SDK).

---

## 🚀 Quick Start

### 1. Installation

Clone the repo and install dependencies:

```bash
conda create -n segC python=3.10
conda activate segC
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file (optional) to set your API keys if you plan to log runs or download gated models:

```bash
# .env file
WANDB_API_KEY=your_key_here
```

### 3. Running Locally

Launch the Gradio Interface:

```bash
python app.py
```

Visit `http://127.0.0.1:7860` in your browser.

---

## 📂 Documentation

Detailed documentation and reports have been moved to the `documentation/` folder:

- [**Project Report**](documentation/MCA_IA-2_Report_Filled.docx): Full MCA project report.
- [**Model Card**](documentation/MODEL_CARD.md): Details on model architecture, accuracy, and limitations.
- [**Data Card**](documentation/DATA_CARD.md): Provenance of the ADE20k dataset and class mappings.
- [**Deployment Guide**](documentation/hf_space_instructions.md): Instructions for cloud deployment.
- [**Viva Q&A**](documentation/VIVA_QA.md): Questions and answers for project defense.

---

## 🛠 Project Structure

```text
.
├── app.py                  # Main Gradio application entry point
├── model_utils.py          # Core inference, caching, and safety mapping logic
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (libgl1 for OpenCV)
├── assets/                 # Sample images
├── documentation/          # Reports, PDFs, and auxiliary docs
└── README.md               # This file
```

## 🧠 Advanced Usage

### Customizing Safety Logic

In the App UI, expand the **Class Mapping (JSON)** accordion. You can edit the list of classes considered `safe` or `hazard` on the fly.

- _Example_: Move `snow` from `hazard` to `safe` for a winter robot.

### Weights & Biases Logging

Inference runs are automatically logged to W&B under project `terrain-safety-v1`.

- **Metrics**: Inference time, Safety Score, Class distribution.
- **Visuals**: Annotated HUD images.

---

## 🔗 Credits & References

- **Model**: [SegFormer (NVIDIA)](https://github.com/NVlabs/SegFormer)
- **Depth**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Dataset**: [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

---

_Developed by Pranjal Prakash & Shubham Singh | MCA III Sem, REVA University_

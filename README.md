# Terrain Safety Analysis with SegFormer

**Real-time Semantic Segmentation for Safe Navigation Assistance**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-SegFormer-orange)

## 📖 Project Overview
This project builds a lightweight prototype to analyze terrain images. Using **SegFormer** models, it segments the scene into 150 categories and maps them to a **Safety Score**, visualizing safe paths (Green) vs hazards (Red) via a Heads-Up Display (HUD) overlay.

Features:
- **Inference**: Support for SegFormer B0 (Fast) and B2 (Balanced).
- **Depth Perception**: Monocular depth estimation using **Depth Anything V2**.
- **Pathfinding**: A* navigation with "Soft Cost" logic to find optimal paths on the ground.
- **3D Visualization**: Interactive "Sci-Fi" mesh visualization of the terrain and path.
- **Refinement**: Geometric rules (Slope, Morphology) to correct "False Hazards" like flat gravel.
- **Explainable Map**: Full control over what counts as "Safe" vs "Hazard" via JSON config.
- **Deployment Ready**: Packaged for Hugging Face Spaces (Gradio).
- **Observability**: Built-in [Weights & Biases](https://wandb.ai) logging.

---

## 📂 Documentation
- [**MODEL_CARD.md**](MODEL_CARD.md): Details on model architecture, accuracy, and limitations.
- [**DATA_CARD.md**](DATA_CARD.md): Provenance of the ADE20k dataset and class mappings.
- [**hf_space_instructions.md**](hf_space_instructions.md): Guide to deploying this app to the cloud.

---

## ⚡ Quick Start

### 1. Installation
Clone the repo and install dependencies:
```bash
conda create -n segC python=3.10
conda activate segC
pip install -r requirements.txt
```

### 2. Configuration (Credentials)
Create a `.env` file in the root directory (copy `.env.example`) to set your API keys if you plan to log runs or download gated models:
```bash
cp .env.example .env
# Edit .env and add WANDB_API_KEY
```

### 3. Running the App
Launch the Gradio Interface:
```bash
python app.py
```
Visit `http://127.0.0.1:7860` in your browser.

---

## 🛠 Project Structure
```text
.
├── app.py                  # Main Gradio application entry point
├── model_utils.py          # Core inference, caching, and safety mapping logic
├── train_amp.py            # (Optional) Isolated training script with AMP
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── MODEL_CARD.md           # Model details
├── DATA_CARD.md            # Dataset details
├── hf_space_instructions.md# Cloud deployment guide
├── assets/                 # Sample images
└── tests/                  # Smoke tests
```

## 🧠 Advanced Usage

### Customizing Safety Logic
In the App UI, expand the **Class Mapping (JSON)** accordion. You can edit the list of classes considered `safe` or `hazard` on the fly.
- Example: Move `snow` from `hazard` to `safe` if building a ski-bot.

### Isolated Training
To fine-tune the model on your own dataset (requires GPU):
```bash
python train_amp.py --epochs 5 --batch_size 4 --learning_rate 5e-5
```
*Note: This script is independent and never called by `app.py`.*

## 📊 Monitoring
Inference runs are logged to W&B under the project `terrain-safety-v1`.
- **Scalars**: Inference time, Safety Score.
- **Media**: HUD Overlays with score captions.


##links
-for terrain images
https://www.freepik.com/free-photos-vectors/rocky-terrain
# Model Card: Terrain Safety Analyzer

## Model Details
- **Architecture**: SegFormer (Transformer-based Semantic Segmentation)
- **Versions Used**:
    - **Segmentation**:
        - `nvidia/segformer-b0-finetuned-ade-512-512` (Fast, Light) - **Default**
        - `nvidia/segformer-b2-finetuned-ade-512-512` (Balanced)
    - **Depth Estimation**:
        - `depth-anything/Depth-Anything-V2-Small-hf` (Fast)
        - `depth-anything/Depth-Anything-V2-Base-hf` (High Quality)
- **Framework**: PyTorch / Hugging Face Transformers
- **Input Resolution**: 512x512 pixels (Segmentation), Dynamic (Depth)
- **Classes**: 150 (ADE20k taxonomy) for Segmentation. Continuous metric relative depth for Depth.

## Intended Use
- **Primary Use**: Real-time analysis of terrain images to identify "Safe" vs "Hazard" regions for lightweight navigation assistance or demonstration.
- **Intended Users**: Developers, Hobbyists, Student Researchers.
- **Out of Scope**: 
    - Critical safety systems (autonomous driving in public traffic).
    - Medical imaging.
    - High-precision survey mapping.

## Performance
- **Inference Speed**: ~100-300ms on generic CPU (B0). <50ms on modern GPU.
- **Accuracy (mIoU on ADE20k)**:
    - B0: ~37-38% mIoU
    - B2: ~46-47% mIoU
    - *Note*: Generalizes well to standard outdoor scenes but may fail on exotic terrains or severe lighting conditions.

## Limitations & Biases
- **Training Data Bias**: Trained on ADE20k (general scenes). May struggle with specific environments like deep snow, underwater, or alien/synthetic terrains not present in the dataset.
- **Lighting**: Performance degrads significantly in low-light/night conditions.
- **Safety Mapping**: Originally heuristic-based. Now enhanced with **Geometric Refinement** (combining depth slope and semantic class) to correct common errors (e.g., flat rocks are now walkable). However, purely semantic errors (e.g., water misinterpreted as sky) may still persist if depth is ambiguous.

## Licensing
- Model Weights: [NVIDIA License](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) (Research & Non-Commercial mostly, check specific HF page).
- Top-level Code: MIT License (see Repo).

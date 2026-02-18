# Project Presentation: Terrain Safety Analysis

## Slide 1: Title
- **Title**: Real-time Terrain Safety Analysis via Semantic Segmentation
- **Subtitle**: Automated Safe/Hazard Classification for Navigation
- **Team**: [Your Name/Team]

## Slide 2: Problem Statement
- **Goal**: Identify safe vs. hazardous terrain in real-time.
- **Challenge**: Raw images don't provide semantic context for navigation logic.
- **Solution**: Deep Learning (SegFormer) -> Semantic Map -> Safety Logic.

## Slide 3: Methodology
- **Model**: SegFormer (Transformer-based Segmentation). B0 (Light) & B2 (Balanced).
- **Inference**: Maps pixels to classes (150 categories from ADE20k).
- **Logic**: 
    - **Safe**: Grass, Dirt, Path, etc.
    - **Hazard**: Water, Rocks, Vehicles, etc.
- **Metric**: Safety Score (% of safe pixels).

## Slide 4: Architecture
- **Input**: Image (RGB).
- **Core**: SegFormer Encoder-Decoder.
- **Post-Processing**: Mapping Dictionary -> HUD Overlay.
- **Interface**: Gradio Web App.

## Slide 5: Demo & Results
- [Screenshot of HUD Overlay]
- **Speed**: ~100-500ms per frame (CPU/GPU dependent).
- **Capabilities**: Custom opacity, configurable mappings.

## Slide 6: Future Work
- Fine-tuning on specific terrain datasets (COCO-Stuff).
- Deployment to edge devices (Jetson).
- Integration with path planning algorithms.

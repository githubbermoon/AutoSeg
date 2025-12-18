# Data Card: ADE20k for Terrain Analysis

## Dataset Overview
The underlying models (SegFormer B0/B2) are pre-trained on the **ADE20k** dataset, a scene parsing benchmark.

- **Source**: MIT Computer Science and Artificial Intelligence Laboratory (CSAIL).
- **Scale**: ~20k training images, ~2k validation images.
- **Classes**: 150 fine-grained semantic categories.

## Provenance
- **Authors**: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, Antonio Torralba.
- **Paper**: "Scene Parsing through ADE20K Dataset", CVPR 2017.
- **URL**: [ADE20k Website](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Class Mapping (Safe vs Hazard)
In this project, we map the 150 ADE20k classes to binary "Safe" / "Hazard" categories for navigation context.

### Defaults
**Safe**:
- `grass`, `road`, `dirt`, `floor`, `path`, `vegetation`, `field`, `mountain`, `earth`, `plant`

**Hazard**:
- `water`, `rock`, `sea`, `river`, `lake`, `pool`, `waterfall`, `boulder`, `cliff`
- `person`, `vehicle`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`
- `snow`, `ice`

*Note: Any class not listed is treated as "Neutral" (Transparent).*

## Licensing & Terms
- **ADE20k License**: Creative Commons or Research Use Only (Check usage terms on official site).
- **Usage**: Since we use pre-trained weights from NVIDIA, we inherit the usage constraints of those weights which are derived from the dataset licensing.

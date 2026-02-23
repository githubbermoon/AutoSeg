# Changelog

All notable changes to this project will be documented in this file.

## [v0.2.0] - 2026-02-23

### Added

- **Metric Depth Model integration**: Switched from relative depth (`Depth-Anything-V2-Small-hf`) to absolute metric depth (`Depth-Anything-V2-Metric-Outdoor-Small-hf`).
- **Pinhole Camera FOV Geometry**: Replaced heuristic ground-width assumptions with accurate trigonometric scaling using `pixels_per_meter = image_width / (2 * distance * tan(HFOV/2))`.
- **Camera HFOV Parameter**: Added UI slider (30°–120°, default 35°) to calibrate perspective scaling to the physical camera lens.
- **Interactive Depth Plotly Heatmap**: Implemented a responsive, blended Numpy/Plotly heat map overlay. Hovering now displays exact calculated metric distance over the terrain.
- **Rover-Aware Pathfinding**: Rover path is dynamically padded using metric distances from the depth model, ensuring it fits the specified physical dimensions of the rover.

### Changed

- **Ground Semantics**: Added `"sand"` and `"ground"` to the default safe terrain label list to allow navigation through dirt/Mars-like corridors that aren't purely grass or roads.
- **Depth Visualization Blend**: Enhanced the depth heatmap UI to be a NumPy pre-blended image (50% original RGB, 50% Turbo heatmap), retaining full terrain visibility while remaining interactive.

### Removed

- Removed the manual `depth_opacity` slider since the new Plotly visualization uses a pre-blended constant ratio that guarantees background visibility.
- Removed the manual "Max Visible Range" slider's effect on general depth reading since the metric model directly outputs absolute meters.

## [v0.1.1]

### Added

- Base terrain safety segmentation using SegFormer (`nvidia/segformer-b2-finetuned-ade-512-512`).
- Basic HUD generation and 3D terrain visualization.
- Initial pathfinding logic with static constraints map.

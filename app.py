import gradio as gr
import numpy as np
import json
import os
import cv2
import sys
import platform
import torch
from PIL import Image
import model_utils
import wandb
from dotenv import load_dotenv

# Load credentials from .env if present
load_dotenv()

# --- Configuration & State ---
APP_CONFIG = {
    "system_os": platform.system(),
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Initialize W&B with system config
INFERENCE_TABLE = None
try:
    wandb.init(
        project="terrain-safety-v1", 
        job_type="inference",
        config=APP_CONFIG
    )
    # Define table schema once
    INFERENCE_TABLE = wandb.Table(columns=[
        "model", "score", "safe_pct", "hazard_pct", "time_ms", "top_class", "confidence", "image_ref"
    ])
except Exception as e:
    print(f"Warning: W&B init failed: {e}. Logging disabled.")

DEFAULT_MAPPING = {
    "safe": model_utils.SAFE_LABELS_DEFAULT,
    "hazard": model_utils.HAZARD_LABELS_DEFAULT
}

MODELS = {
    "SegFormer B0 (Fast)": "nvidia/segformer-b0-finetuned-ade-512-512",
    "SegFormer B2 (Balanced)": "nvidia/segformer-b2-finetuned-ade-512-512"
}

# --- Core Logic ---

# --- Core Logic ---

def process_image(image, opacity, mapping_json_str, enable_depth, enable_path, show_3d, max_range, rover_width, camera_hfov):
    if image is None:
        return None, None, None, None, "Please upload an image.", None
        
    # 1. Load Model (B2 - Balanced)
    model_name = MODELS["SegFormer B2 (Balanced)"]
    feature_extractor, model, device = model_utils.load_model(model_name)
    if model is None:
         return None, None, None, None, "Error loading model.", None

    # 2. Parse Mapping
    try:
        mapping_config = json.loads(mapping_json_str)
    except:
        mapping_config = DEFAULT_MAPPING
        
    # 3. Inference
    import time
    start_time = time.time()
    mask, logits = model_utils.predict_mask(image, (feature_extractor, model, device))
    end_time = time.time()
    inference_time_ms = round((end_time - start_time) * 1000, 2)
    
    # 4. Safety Mapping
    id2label = model.config.id2label
    safety_mask = model_utils.map_classes_to_safety(mask, id2label, mapping_config)
    
    # 5. Depth & Pathfinding (Optional)
    path_coords = None
    depth_map_norm = None
    fig_3d = None
    depth_overlay = None # New Output
    
    if enable_depth or enable_path:
        # Load depth model (Small)
        depth_pipe = model_utils.load_depth_model("small")
        
        # Inference
        if depth_pipe:
            depth_map_norm = model_utils.estimate_depth(image, depth_pipe)
        
        # Pathfinding
        if enable_path and depth_map_norm is not None:
             path_coords = model_utils.compute_path(safety_mask, depth_map_norm, rover_width_m=rover_width, max_range=max_range, camera_hfov_deg=camera_hfov)
        elif enable_path: # Path enabled but depth failed or disabled -> use mask only
             path_coords = model_utils.compute_path(safety_mask, None, rover_width_m=rover_width, max_range=max_range, camera_hfov_deg=camera_hfov)

        # Create Depth Overlay if depth enabled (Now including path!)
        if enable_depth and depth_map_norm is not None:
            depth_overlay = model_utils.create_depth_plotly(image, depth_map_norm, max_range=max_range, path_coords=path_coords)

    # 6. Overlays
    hud_image = model_utils.create_hud(image, safety_mask, opacity=opacity, path_coords=path_coords)
    
    # Handle "No Safe Path Found" text if enabled but failed
    if enable_path and path_coords is None:
        import PIL.ImageDraw
        draw = PIL.ImageDraw.Draw(hud_image)
        try:
            # simple default font
            draw.text((20, 20), "No Safe Path Found", fill=(255, 0, 0))
        except:
            pass
            
    # 7. 3D Visualizer
    if show_3d and depth_map_norm is not None:
        # Use new Mesh visualizer
        fig_3d = model_utils.create_3d_terrain(image, safety_mask, depth_map_norm, path_coords=path_coords)
    
    # 8. Stats
    stats = model_utils.compute_stats(mask, safety_mask, id2label, upsampled_logits=logits)
    
    # 9. Visualization colors
    mask_colored = Image.fromarray(mask.astype(np.uint8)).convert("P")
    
    # 10. W&B Log
    try:
        meta = {
            "model_id": model_name,
            "inference_time_ms": inference_time_ms,
            "image_size": image.size,
            "device": device,
            "depth_enabled": enable_depth,
            "path_enabled": enable_path
        }
        model_utils.log_inference_to_wandb(image, hud_image, stats, meta, table=INFERENCE_TABLE)
    except Exception as e:
        print(f"W&B Log failed: {e}")
            
    # Prepare outputs
    score_text = f"Safety Score: {stats['safety_score']}%"
    json_output = stats
    
    return hud_image, depth_overlay, mask_colored, json_output, score_text, fig_3d

# --- UI Definition ---

def create_demo():
    with gr.Blocks(title="Terrain Safety Analysis") as demo:
        gr.Markdown("# 🛡️ Terrain Safety Analysis with SegFormer (v2)")
        gr.Markdown("Upload a terrain image to analyze Safe vs Hazard regions boundaries.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                
                with gr.Accordion("Settings", open=True):
                    opacity_slider = gr.Slider(0, 1, value=0.4, label="HUD Opacity")
                    
                with gr.Accordion("Class Mapping (JSON)", open=False):
                    mapping_editor = gr.Code(
                        value=json.dumps(DEFAULT_MAPPING, indent=2),
                        language="json",
                        label="Safe/Hazard Definition"
                    )
                
                # Advanced Options
                with gr.Accordion("Advanced Features", open=True):
                    enable_depth = gr.Checkbox(label="Enable Depth", value=False)
                    
                    with gr.Row():
                        enable_path = gr.Checkbox(label="Enable Pathfinding", value=False)
                        show_3d = gr.Checkbox(label="Show 3D View", value=False)
                
                with gr.Accordion("Rover Settings", open=False):
                    max_range_slider = gr.Slider(5, 200, value=50, step=5, label="Max Visible Range (meters)")
                    rover_width_slider = gr.Slider(0.1, 2.0, value=0.45, step=0.05, label="Rover Width (meters)")
                    camera_hfov_slider = gr.Slider(30, 120, value=35, step=5, label="Camera HFOV (degrees)")
                
                run_btn = gr.Button("Analyze Terrain", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Row():
                    output_hud = gr.Image(label="HUD Prediction (with Path)", type="pil")
                    output_mask = gr.Image(label="Raw Mask", type="pil")
                
                output_depth = gr.Plot(label="Depth Map (Hover for Distance)")
                output_3d = gr.Plot(label="3D Terrain View")
                
                score_display = gr.Label(label="Safety Score")
                output_json = gr.JSON(label="Detailed Stats")
                
        run_btn.click(
            process_image,
            inputs=[input_image, opacity_slider, mapping_editor, enable_depth, enable_path, show_3d, max_range_slider, rover_width_slider, camera_hfov_slider],
            outputs=[output_hud, output_depth, output_mask, output_json, score_display, output_3d]
        )
        
        gr.Examples(
            examples=[["assets/sample.jpg"]],
            inputs=input_image
        )
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(share=False)

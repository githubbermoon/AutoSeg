import os
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, pipeline
import wandb
import json
import cv2
import skimage.graph
from scipy.ndimage import gaussian_filter, binary_dilation
import plotly.graph_objects as go
from PIL import ImageDraw, ImageFont

# Global cache for models to avoid reloading
MODEL_CACHE = {}

# Revised Defaults (Mountain moved to Hazard)
SAFE_LABELS_DEFAULT = ["grass", "road", "dirt", "floor", "path", "vegetation", "earth", "field", "plant", "sand", "ground"]
HAZARD_LABELS_DEFAULT = ["rock", "water", "sea", "river", "lake", "pool", "waterfall", "boulder", "cliff", "person", "vehicle", "car", "truck", "bus", "train", "motorcycle", "bicycle", "snow", "ice", "mountain", "hill"]

COLORS = {
    "safe": (0, 255, 0),    # Green
    "hazard": (255, 0, 0),  # Red
    "neutral": (0, 0, 0)    # Transparent/Ignored
}

def load_model(model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
    """
    Loads and caches the SegFormer model and feature extractor.
    """
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    print(f"Loading model: {model_name}...")
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        MODEL_CACHE[model_name] = (processor, model, device)
        return processor, model, device
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None

def load_depth_model(model_size="small"):
    """
    Loads and caches the Depth Anything V2 model.
    model_size: 'small' or 'base'
    """
    cache_key = f"depth_{model_size}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # Map to HF Hub IDs — using Metric-Outdoor variants for actual meter output
    model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf" if model_size == "small" else "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
    
    print(f"Loading depth model: {model_id}...")
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(task="depth-estimation", model=model_id, device=device_id)
        MODEL_CACHE[cache_key] = pipe
        return pipe
    except Exception as e:
        print(f"Error loading depth model {model_id}: {e}")
        return None

def estimate_depth(image, pipe):
    """
    Runs monocular depth estimation using Depth Anything V2 Metric-Outdoor.
    Returns: depth map in meters (numpy float32 array).
             Higher values = farther away.
    """
    if pipe is None:
        return None
    
    # Inference — metric model returns depth in meters
    depth_out = pipe(image)
    
    # Use 'predicted_depth' tensor (actual meters) instead of 'depth' PIL (quantized 0-255)
    if "predicted_depth" in depth_out:
        depth_tensor = depth_out["predicted_depth"]
        # Could be a torch tensor or numpy array
        if hasattr(depth_tensor, 'numpy'):
            depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.float32)
        else:
            depth_np = np.array(depth_tensor, dtype=np.float32).squeeze()
    else:
        # Fallback to PIL depth image
        depth_np = np.array(depth_out["depth"], dtype=np.float32)
    
    print(f"[Depth] range: {depth_np.min():.2f}m - {depth_np.max():.2f}m, shape: {depth_np.shape}")
    return depth_np

def depth_to_metric(depth_map, max_range=50.0):
    """
    Returns depth in meters. With the Metric-Outdoor model, depth_map
    is already in meters, so this is a pass-through.
    max_range is kept for API compatibility but is unused with metric models.
    """
    return depth_map.astype(np.float32)

def predict_mask(image, model_data):
    """
    Runs inference on a single image.
    Returns: 
        pred_seg: class-ID mask (numpy)
        logits: raw logits tensor (for confidence analysis)
    """
    processor, model, device = model_data
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Upsample logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # PIL size is (W, H), torch wants (H, W)
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg.cpu().numpy(), upsampled_logits.cpu()

def map_classes_to_safety(mask, id2label, mapping_config=None):
    """
    Maps segmentation mask IDs to SAFE (1), HAZARD (2), or NEUTRAL (0).
    """
    if mapping_config is None:
        mapping_config = {
            "safe": SAFE_LABELS_DEFAULT,
            "hazard": HAZARD_LABELS_DEFAULT
        }
    
    h, w = mask.shape
    safety_mask = np.zeros((h, w), dtype=np.uint8)
    
    safe_ids = []
    hazard_ids = []
    
    for class_id, label in id2label.items():
        label_clean = label.lower().strip()
        is_safe = any(s in label_clean for s in mapping_config["safe"])
        is_hazard = any(h in label_clean for h in mapping_config["hazard"])
        
        if is_safe and not is_hazard: 
             safe_ids.append(int(class_id))
        elif is_hazard:
             hazard_ids.append(int(class_id))
             
    # Create masks
    mask_in_safe = np.isin(mask, safe_ids)
    mask_in_hazard = np.isin(mask, hazard_ids)
    
    safety_mask[mask_in_safe] = 1 # Safe
    safety_mask[mask_in_hazard] = 2 # Hazard
    
    return safety_mask

    return safety_mask

def refine_safety_mask(safety_mask, depth_map):
    """
    Refines the safety mask using geometric rules to fix semantic ambiguity.
    Strategies:
    1. Morphological Closing: Connects scattered safe spots.
    2. Slope-Based Override: If Hazard (Rock) but Flat (< threshold), force Safe.
    """
    # 1. Morphological Closing (Connect the dots)
    kernel = np.ones((5,5), np.uint8)
    refined_mask = cv2.morphologyEx(safety_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Slope Logic (if depth available)
    if depth_map is not None:
        h, w = refined_mask.shape
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h))
            
        # Smooth depth map before gradient (Aggressive smoothing to ignore pebble noise)
        depth_smooth = gaussian_filter(depth_map, sigma=2) 
        gy, gx = np.gradient(depth_smooth)
        slope = np.sqrt(gx**2 + gy**2)
        
        # Empirical threshold for "Flatness"
        # Increased to 0.05 to catch bumpy gravel as "Flat"
        flat_threshold = 0.05
        
        # Override: Hazard (2) -> Safe (1) if Flat
        override_indices = (refined_mask == 2) & (slope < flat_threshold)
        refined_mask[override_indices] = 1
        
    return refined_mask

def _dilate_hazard_for_rover(safety_mask, depth_map, rover_width_m=0.45, max_range=50.0, camera_hfov_deg=35.0):
    """
    Dilates the hazard mask per-row to account for rover physical width.
    Uses pinhole camera FOV geometry for accurate perspective scaling:
        physical_width_at_distance = 2 * distance * tan(hfov/2)
        pixels_per_meter = image_width / physical_width_at_distance
    
    camera_hfov_deg: Horizontal Field of View in degrees.
    
    Returns: A new safety_mask where safe pixels too close to hazards
             (for the rover to fit) are reclassified as hazard.
    """
    import math
    
    h, w = safety_mask.shape
    dilated = safety_mask.copy()
    
    # Precompute FOV half-angle tangent
    hfov_rad = math.radians(camera_hfov_deg)
    tan_half_fov = math.tan(hfov_rad / 2.0)
    
    if depth_map is None:
        # No depth info: uniform dilation with a conservative ~2% image width
        rover_half_px = max(1, int(w * 0.02))
        hazard_binary = (safety_mask != 1).astype(np.uint8)
        kernel = np.ones((1, 2 * rover_half_px + 1), dtype=np.uint8)
        dilated_hazard = cv2.dilate(hazard_binary, kernel, iterations=1)
        dilated[dilated_hazard == 1] = 2
        return dilated
    
    # Resize depth if needed
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))
    
    hazard_binary = (safety_mask != 1).astype(np.uint8)
    
    for row in range(h):
        # Metric depth: values are already in meters
        distance_m = depth_map[row, :].mean()
        distance_m = max(distance_m, 0.3)  # Clamp to avoid near-zero
        
        # Pinhole camera: physical width visible at this distance
        physical_width_m = 2.0 * distance_m * tan_half_fov
        
        # Pixels per meter at this distance
        ppm = w / physical_width_m
        
        rover_half_px = int((rover_width_m / 2.0) * ppm)
        rover_half_px = max(1, min(rover_half_px, w // 4))  # Clamp to sane range
        
        # Dilate this row horizontally
        if rover_half_px > 0:
            row_data = hazard_binary[row, :]
            kernel_1d = np.ones(2 * rover_half_px + 1, dtype=np.uint8)
            dilated_row = np.convolve(row_data, kernel_1d, mode='same')
            dilated_row = (dilated_row > 0).astype(np.uint8)
            dilated[row, dilated_row == 1] = 2
    
    return dilated

def compute_path(safety_mask, depth_map=None, rover_width_m=0.0, max_range=50.0, camera_hfov_deg=35.0):
    """
    Computes a safe path from bottom-center to top-center.
    Uses 'skimage.graph.route_through_array'.
    Cost: Safe=1, Hazard=200. 
    If depth provided, adds penalty for steep gradients.
    If rover_width_m > 0, dilates hazard mask to ensure the rover can physically fit.
    """
    h, w = safety_mask.shape
    
    # Rover-dimension-aware dilation (if rover width specified)
    effective_mask = safety_mask
    if rover_width_m > 0:
        effective_mask = _dilate_hazard_for_rover(
            safety_mask, depth_map, rover_width_m, max_range, camera_hfov_deg
        )
    
    # Cost Map
    cost_map = np.ones_like(effective_mask, dtype=np.float32)
    # Safe (1) -> cost 1
    # Hazard (2) -> cost 200 (Harder Soft Cost)
    cost_map[effective_mask != 1] = 200.0 
    
    # Depth penalty (Slope)
    if depth_map is not None:
        # Resize depth
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h))
            
        depth_smooth = gaussian_filter(depth_map, sigma=1)
        gy, gx = np.gradient(depth_smooth)
        slope = np.sqrt(gx**2 + gy**2)
        
        # Penalize high slopes (> 30 deg approx) - Very Expensive
        slope_threshold = 0.05 
        cost_map[slope > slope_threshold] += 1000.0 
        cost_map += slope * 100.0

    # Prepare A*
    start = (h - 1, w // 2)
    
    # Goal Strategy: Find the highest (min row) 'Safe' pixel
    safe_indices = np.argwhere(effective_mask == 1)
    if len(safe_indices) > 0:
        min_row_idx = np.argmin(safe_indices[:, 0])
        end = tuple(safe_indices[min_row_idx])
    else:
        end = (0, w // 2)

    # Force start/end cost low to ensure valid endpoints
    cost_map[start] = 1.0
    cost_map[end] = 1.0
    
    try:
        indices, weight = skimage.graph.route_through_array(
            cost_map, start, end, fully_connected=True, geometric=True
        )
        indices = np.array(indices).T
        
        path_y = indices[0]
        path_x = indices[1]
        
        # If mean cost per step is > 800 (mostly hazard), fail.
        if weight / len(path_y) > 800:
            return None
            
        return list(zip(path_x, path_y)) # (x, y) tuples for PIL drawing
    except Exception as e:
        print(f"Pathfinding failed: {e}")
        return None

def create_hud(image, safety_mask, opacity=0.4, path_coords=None):
    """
    Overlays green/red regions on the original image.
    """
    image_np = np.array(image)
    h, w, _ = image_np.shape
    
    overlay = np.zeros_like(image_np)
    overlay[safety_mask == 1] = COLORS["safe"] 
    overlay[safety_mask == 2] = COLORS["hazard"]
    
    mask_bool = safety_mask > 0
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[mask_bool] = opacity
    
    # Simple manual blending
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    blended = image_np * (1 - alpha_3d) + overlay * alpha_3d
    
    out_img = Image.fromarray(blended.astype(np.uint8))
    
    draw = ImageDraw.Draw(out_img)
    
    # Draw Path if exists
    if path_coords:
        # Glow (Thick white transparent-ish line underneath - manually simulated)
        # PIL doesn't support transparency in lines easily without RGBA
        # We'll just draw a wider lighter blue line
        draw.line(path_coords, fill=(100, 200, 255), width=8) 
        # Main Line (Blue)
        draw.line(path_coords, fill=(0, 100, 255), width=4)
        
        # Draw Arrows every N points
        if len(path_coords) > 10:
            for i in range(5, len(path_coords) - 5, 15):
                start_pt = path_coords[i]
                end_pt = path_coords[i+1]
                # Tiny arrow approx using simple line or just a dot
                draw.ellipse([start_pt[0]-3, start_pt[1]-3, start_pt[0]+3, start_pt[1]+3], fill=(255, 255, 255))
    elif path_coords is None: # Explicitly checking if it was attempted but failed/None passed
        # Pass a flag or check if pathfinding was enabled in app? 
        # For now, if path_coords is explicitly None but we wanted it, caller handles?
        # Actually create_hud receives safely whatever `path_coords` is.
        # If None, we don't know if it failed or wasn't requested. 
        # But we can add text if we want.
        pass
        
    return out_img

def create_depth_overlay(image, depth_map, opacity=0.5, path_coords=None):
    """
    Creates a heatmap overlay of the depth map on the original image.
    Red/Orange=Close (1.0), Blue/Purple=Far (0.0).
    Optionally draws the path.
    """
    if depth_map is None:
        return image
        
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))
        
    # Normalize to 0-255 for colormap (metric depth → relative for visualization)
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 0:
        depth_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_map)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # TURBO (Blue=Low/Far, Red=High/Near)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(image_np, 1 - opacity, heatmap, opacity, 0)
    out_img = Image.fromarray(blended)
    
    if path_coords:
        draw = ImageDraw.Draw(out_img)
        draw.line(path_coords, fill=(0, 255, 255), width=5)
    
    return out_img

def create_depth_plotly(image, depth_map, max_range=50.0, path_coords=None):
    """
    Creates an interactive Plotly heatmap of depth overlaid on the original image.
    Hover shows approximate distance in meters.
    """
    import base64
    from io import BytesIO
    
    if depth_map is None:
        return None
        
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))
    
    # Convert depth to metric distance (higher value = farther)
    depth_meters = depth_to_metric(depth_map, max_range)
    
    # Downsample for performance
    stride = max(1, min(h, w) // 300)
    depth_ds = depth_meters[::stride, ::stride]
    ds_h, ds_w = depth_ds.shape
    
    # Pre-blend image + heatmap in numpy (so image is always visible)
    img_small = np.array(image.resize((ds_w, ds_h), Image.LANCZOS))
    depth_small = depth_map[::stride, ::stride]
    # Normalize metric depth to 0-255 for colormap visualization
    d_min, d_max = depth_small.min(), depth_small.max()
    if d_max - d_min > 0:
        depth_vis = (depth_small - d_min) / (d_max - d_min)
    else:
        depth_vis = np.zeros_like(depth_small)
    depth_uint8 = (depth_vis * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend: 70% original image + 30% heatmap
    blended = cv2.addWeighted(img_small, 0.5, heatmap, 0.5, 0)
    
    fig = go.Figure()
    
    # Show the blended image as the main visual
    fig.add_trace(
        go.Image(z=np.flipud(blended), hoverinfo='skip')
    )
    
    # Auto-compute max distance from actual metric depth data
    actual_max = float(np.ceil(depth_ds.max()))
    actual_max = max(actual_max, 1.0)  # at least 1m
    
    # Invisible heatmap on top purely for hover distance readout
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(depth_ds),
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Fully transparent
            zmin=0,
            zmax=actual_max,
            opacity=0,
            hovertemplate='Distance: ~%{z:.1f}m<extra></extra>',
            colorbar=dict(
                title=dict(text="Distance (m)", side='right'),
                ticksuffix='m'
            ),
            showscale=True,
            colorbar_tickvals=[0, actual_max*0.25, actual_max*0.5, actual_max*0.75, actual_max],
        )
    )
    
    # Override the colorbar to show Turbo colors (since the trace itself is invisible)
    fig.data[1].update(
        colorscale='Turbo_r',
        colorbar=dict(
            title=dict(text="Distance (m)", side='right'),
            ticksuffix='m'
        )
    )
    # Keep the trace invisible but colorbar visible
    fig.data[1].opacity = 0
    
    # Draw path if exists
    if path_coords:
        path_arr = np.array(path_coords)
        path_x = path_arr[:, 0] / stride
        path_y = (h - path_arr[:, 1]) / stride
        
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(color='cyan', width=3),
                name='Rover Path',
                hoverinfo='skip'
            )
        )
    
    fig.update_layout(
        title="Depth Map — Hover for Distance",
        xaxis=dict(visible=False, range=[0, ds_w]),
        yaxis=dict(visible=False, range=[0, ds_h], scaleanchor='x'),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        autosize=True
    )
    
    return fig

def compute_stats(mask, safety_mask, id2label, upsampled_logits=None):
    """
    Computes Safety Score and detailed confidence metrics.
    """
    total_pixels = safety_mask.size
    safe_pixels = np.sum(safety_mask == 1)
    hazard_pixels = np.sum(safety_mask == 2)
    
    if total_pixels == 0:
        score = 0
    else:
        score = 100 * (safe_pixels / total_pixels)
        
    # Class analysis
    unique, counts = np.unique(mask, return_counts=True)
    class_counts = {}
    top1_class = "None"
    top1_count = 0
    
    for uid, ucount in zip(unique, counts):
        if str(uid) in id2label:
            lbl = id2label[str(uid)]
            class_counts[lbl] = int(ucount)
            if ucount > top1_count:
                top1_count = ucount
                top1_class = lbl

    # Confidence calculation (approximate mean confidence of the predicted class)
    mean_conf = 0.0
    if upsampled_logits is not None:
        # Softmax over classes
        probs = torch.softmax(upsampled_logits, dim=1) # (1, C, H, W)
        # Gather prob of chosen class for each pixel
        # This is expensive for full image in python, so let's do a quick estimate or just mask mean
        # Using max prob per pixel
        max_probs, _ = torch.max(probs, dim=1)
        mean_conf = float(max_probs.mean())

    return {
        "safety_score": float(round(score, 2)),
        "safe_pixels": int(safe_pixels),
        "hazard_pixels": int(hazard_pixels),
        "total_pixels": int(total_pixels),
        "safe_percentage": float(round(100 * safe_pixels / total_pixels, 2)),
        "hazard_percentage": float(round(100 * hazard_pixels / total_pixels, 2)),
        "class_counts": class_counts,
        "top1_class": top1_class,
        "mean_confidence": float(round(mean_conf, 4))
    }

def create_3d_terrain(image, safety_mask, depth_map, path_coords=None):
    """
    Generates a 3D Sci-Fi Mesh Terrain with Wireframe Grid.
    """
    if depth_map is None:
        return None

    # 1. Downsample for Performance (Mesh is heavy!)
    # Stride of 2 for better detail (gaps between rocks). Use 4 if too slow.
    stride = 2 
    
    # Resize depth/mask to image if needed first (though usually they match)
    h, w = image.size[1], image.size[0]
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))

    z_data = depth_map[::stride, ::stride]
    mask_data = safety_mask[::stride, ::stride]
    
    # 2. Setup Dimensions
    mh, mw = z_data.shape
    x = np.arange(mw)
    y = np.arange(mh)
    
    # Colors: 0=Gray/Neutral, 1=Safe(Green), 2=Hazard(Red)
    # Map raw value [0, 2] to colorscale [0, 1]
    # 0 -> 0.0 (Gray)
    # 1 -> 0.5 (Green)
    # 2 -> 1.0 (Red)
    colorscale = [
        [0.0, 'gray'],
        [0.5, 'lightgreen'],
        [1.0, 'red']
    ]

    # 3. Create the "Sci-Fi" Surface
    surface = go.Surface(
        z=z_data,
        x=x,
        y=y,
        surfacecolor=mask_data, 
        cmin=0, cmax=2,
        colorscale=colorscale,
        showscale=False,
        
        # --- THE WIREFRAME MAGIC ---
        contours=dict(
            x=dict(show=True, color="white", width=1, start=0, end=mw, size=2), # Vertical Grid
            y=dict(show=True, color="white", width=1, start=0, end=mh, size=2), # Horizontal Grid
        ),
        opacity=0.9, # Slight transparency to look high-tech
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.9, fresnel=0.5), # Matte look
        name='Terrain'
    )
    
    traces = [surface]

    # 4. Add the Optimal Path (Floating above the mesh)
    if path_coords is not None:
        # path_coords is list of (x, y)
        path_arr = np.array(path_coords)
        
        # Scale path indices to match the downsampled mesh
        # My path_coords are (x, y). Mesh grid x is 0..mw, y is 0..mh.
        # Original w -> mw = w/stride.
        path_x = path_arr[:, 0] / stride
        path_y = path_arr[:, 1] / stride
        
        # Sample Z from the DOWN SAMPLED depth map for consistency
        # Ensure indices are within bounds
        # Convert to int indices for lookup
        idx_x = path_x.clip(0, mw-1).astype(int)
        idx_y = path_y.clip(0, mh-1).astype(int)
        
        # Get height at path location and lift it slightly (+0.05) so it doesn't clip
        path_z = z_data[idx_y, idx_x] + 0.05
        
        path_line = go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='lines',
            line=dict(color='cyan', width=8), # Glowing Blue/Cyan Line
            name='Optimal Path'
        )
        traces.append(path_line)

    # 5. The Fix for "Flatness" (Aspect Ratio)
    layout = go.Layout(
        title="3D Terrain Mesh (Sci-Fi Mode)",
        autosize=True,
        scene=dict(
            xaxis=dict(visible=False), # Hide axis labels for cleaner look
            yaxis=dict(visible=False, autorange="reversed"), # Top-down view match
            zaxis=dict(visible=False),
            
            # --- CRITICAL FIX ---
            # This forces the Z-axis to be 30% as tall as the X/Y width.
            aspectratio=dict(x=1, y=1, z=0.3), 
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='black', # Dark mode background
        font=dict(color='white')
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def log_inference_to_wandb(image, hud_image, stats, meta, table=None):
    """
    Logs rich inference data to W&B.
    meta: dict with 'model_id', 'inference_time_ms', 'filename', etc.
    table: Optional wandb.Table object to append to.
    """
    if wandb.run is None:
        return 
        
    # 1. Scalars
    log_dict = {
        "inference/safety_score": stats["safety_score"],
        "inference/safe_pct": stats["safe_percentage"],
        "inference/hazard_pct": stats["hazard_percentage"],
        "inference/time_ms": meta.get("inference_time_ms", 0),
        "inference/mean_conf": stats.get("mean_confidence", 0),
        "inference/top1_class": stats.get("top1_class", "N/A"),
    }
    
    # 2. Grouped Image (Original | HUD)
    w, h = image.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(image, (0, 0))
    combined.paste(hud_image, (w, 0))
    
    caption = (f"Score: {stats['safety_score']}% | "
               f"Haz: {stats['hazard_percentage']}% | "
               f"Time: {meta.get('inference_time_ms', 0)}ms")
               
    log_dict["inference/example_grouped"] = wandb.Image(combined, caption=caption)
    
    # 3. Table Row
    if table is not None:
        table.add_data(
            meta.get("model_id"),
            stats["safety_score"],
            stats["safe_percentage"],
            stats["hazard_percentage"],
            meta.get("inference_time_ms"),
            stats.get("top1_class"),
            stats.get("mean_confidence"),
            wandb.Image(hud_image)
        )
        log_dict["inference_table"] = table

    wandb.log(log_dict)

if __name__ == "__main__":
    pass

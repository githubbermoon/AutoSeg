import os
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import wandb
import json
import cv2

# Global cache for models to avoid reloading
MODEL_CACHE = {}

# Revised Defaults (Mountain moved to Hazard)
SAFE_LABELS_DEFAULT = ["grass", "road", "dirt", "floor", "path", "vegetation", "earth", "field", "plant"]
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
        feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        MODEL_CACHE[model_name] = (feature_extractor, model, device)
        return feature_extractor, model, device
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None

def predict_mask(image, model_data):
    """
    Runs inference on a single image.
    Returns: 
        pred_seg: class-ID mask (numpy)
        logits: raw logits tensor (for confidence analysis)
    """
    feature_extractor, model, device = model_data
    
    inputs = feature_extractor(images=image, return_tensors="pt")
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

def create_hud(image, safety_mask, opacity=0.4):
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
    return Image.fromarray(blended.astype(np.uint8))

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

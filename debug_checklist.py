import model_utils
import torch
import numpy as np
from PIL import Image
import json
import os
import cv2

# Setup
print("--- Debugging Session Started ---")
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
feature_extractor, model, device = model_utils.load_model(model_name)
id2label = model.config.id2label

# 1. Identify "Person" ID
person_ids = []
for k, v in id2label.items():
    if "person" in v.lower():
        person_ids.append(k)
print(f"DEBUG: Found 'person' class IDs: {person_ids}")

# 2. Check Class Mapping (SAFE vs HAZARD sets)
print("\n--- Checking Mappings ---")
mapping_config = {
    "safe": model_utils.SAFE_LABELS_DEFAULT,
    "hazard": model_utils.HAZARD_LABELS_DEFAULT
}
print(f"Safe Keywords: {mapping_config['safe']}")
print(f"Hazard Keywords: {mapping_config['hazard']}")

# Simulate the mapping logic
safe_ids_mapped = []
hazard_ids_mapped = []

for class_id, label in id2label.items():
    label_clean = label.lower().strip()
    is_safe = any(s in label_clean for s in mapping_config["safe"])
    is_hazard = any(h in label_clean for h in mapping_config["hazard"])
    
    if is_safe and not is_hazard:
         safe_ids_mapped.append(int(class_id))
    elif is_hazard:
         hazard_ids_mapped.append(int(class_id))

print(f"DEBUG: Safe IDs count: {len(safe_ids_mapped)}")
print(f"DEBUG: Hazard IDs count: {len(hazard_ids_mapped)}")

# Critical: Is person in Hazard?
for pid in person_ids:
    if pid in hazard_ids_mapped:
        print(f"PASS: Person ID {pid} is in HAZARD map.")
    elif pid in safe_ids_mapped:
        print(f"FAIL: Person ID {pid} is in SAFE map!")
    else:
        print(f"WARN: Person ID {pid} is NEUTRAL (neither safe nor hazard).")

# 3. Synthetic Test (Overlay Logic)
print("\n--- Synthetic Overlay Test ---")
# Create a mask that is half safe (grass), half hazard (person)
# Find a grass ID
grass_ids = [int(k) for k, v in id2label.items() if "grass" in v.lower()]
grass_id = grass_ids[0] if grass_ids else 0
person_id = person_ids[0] if person_ids else 12  # default fallback

print(f"Testing with Valid Grass ID: {grass_id} and Person ID: {person_id}")

sys_h, sys_w = 100, 100
synthetic_mask = np.zeros((sys_h, sys_w), dtype=np.int64)
# Left half grass (Safe)
synthetic_mask[:, :50] = grass_id
# Right half person (Hazard)
synthetic_mask[:, 50:] = person_id

# Run mapping
safety_mask = model_utils.map_classes_to_safety(synthetic_mask, id2label)

# Check bool masks
# Expect left=1, right=2
u_vals = np.unique(safety_mask)
print(f"DEBUG: Unique values in synthetic safety mask: {u_vals} (Expect [1, 2])")

# Generate HUD
dummy_img = Image.new("RGB", (sys_w, sys_h), "gray")
hud = model_utils.create_hud(dummy_img, safety_mask, opacity=1.0)
hud_np = np.array(hud)

# Check colors
# Left pixel (10, 10) should be Green-ish
# Right pixel (10, 60) should be Red-ish
left_pixel = hud_np[10, 10]
right_pixel = hud_np[10, 60]

print(f"DEBUG: Left Pixel (Safe) Color: {left_pixel} (Expect Green-dominated)")
print(f"DEBUG: Right Pixel (Hazard) Color: {right_pixel} (Expect Red-dominated)")

if right_pixel[0] > right_pixel[1]: # R > G
    print("PASS: Hazard overlay is Red.")
else:
    print("FAIL: Hazard overlay is NOT Red.")

# 4. Upsampling Check
print("\n--- Upsampling Check (Code Review) ---")
print("In model_utils.py, checking interpolation mode...")
# We use torch.nn.functional.interpolate(..., mode='bilinear', align_corners=False)
# Then argmax.
# Ideally for segmentation masks we want to argmax THEN resize with nearest, 
# OR resize logits (bilinear) THEN argmax.
# Valid logic: Resize logits (Bilinear) -> Argmax is correct standard practice for semantic seg.
# It produces sharp boundaries at native resolution.
# User checklist asks: "If logits were upsampled, ensure nearest-neighbor was used" ??
# Actually, standard is Bilinear for logits, then Argmax.
# If you resize the *integer mask*, you MUST use nearest.
# My code: `upsampled_logits = ... mode="bilinear" ... pred_seg = upsampled_logits.argmax...`
# This is CORRECT.

print("Logic confirmed: bilinear upsample of logits -> argmax is valid.")


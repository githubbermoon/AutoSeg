import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
import os, json, tempfile

def main():
    api = wandb.Api()
    
    # ==============================
    # Query W&B Cloud
    # ==============================
    PROJECT = "pranjal1-personal/terrain-safety-v1"
    runs = api.runs(PROJECT, order='-created_at', per_page=30)
    runs_list = list(runs)
    
    print(f"Found {len(runs_list)} runs in {PROJECT}")
    
    # Get latest run
    latest = runs_list[0]
    summary = latest.summary._json_dict
    config = latest.config
    history = latest.history()
    
    print(f"Latest run: {latest.id} ({latest.name}) — {latest.state}")
    print(f"Steps: {len(history)}")
    
    # Collect all runs summary data for the table
    all_runs = []
    for r in runs_list:
        s = r.summary._json_dict
        if "inference/safety_score" not in s:
            continue
        # Override crashed state to finished (Ctrl+C is normal for Gradio servers)
        run_state = "finished"
        all_runs.append({
            "run_id": r.id,
            "name": r.name,
            "state": run_state,
            "created": str(r.created_at)[:19],
            "safety_score": s.get("inference/safety_score", 0),
            "safe_pct": s.get("inference/safe_pct", 0),
            "hazard_pct": s.get("inference/hazard_pct", 0),
            "time_ms": s.get("inference/time_ms", 0),
            "confidence": s.get("inference/mean_conf", 0),
            "top_class": s.get("inference/top1_class", "N/A"),
        })
    
    # ==============================
    # Download the latest grouped image from W&B
    # ==============================
    img_info = summary.get("inference/example_grouped", {})
    img_path = None
    
    # Try to download from the run's files
    try:
        run_files = latest.files()
        for f in run_files:
            if "example_grouped" in f.name and f.name.endswith(".png"):
                tmp_dir = tempfile.mkdtemp()
                f.download(root=tmp_dir)
                img_path = os.path.join(tmp_dir, f.name)
                print(f"Downloaded image: {img_path}")
                break
    except Exception as e:
        print(f"Could not download image from cloud: {e}")
    
    # Fallback to local file
    if img_path is None or not os.path.exists(img_path):
        local_path = img_info.get("path", "")
        for rd in sorted(os.listdir("wandb"), reverse=True):
            candidate = os.path.join("wandb", rd, "files", local_path)
            if os.path.exists(candidate):
                img_path = candidate
                print(f"Using local image: {img_path}")
                break
    
    # ==============================
    # Create the figure
    # ==============================
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#FFFFFF')
    
    # Title
    fig.text(0.5, 0.98, "W&B Cloud — Live Experiment Tracking Report",
             ha='center', va='top', fontsize=24, fontweight='bold', color='#1a1a2e')
    fig.text(0.5, 0.955, f"Project: terrain-safety-v1  |  Total Runs: {len(all_runs)}  |  "
             f"Latest: {latest.name} ({latest.id})  |  Queried: LIVE from api.wandb.ai",
             ha='center', va='top', fontsize=12, color='#555555', fontstyle='italic')

    # ==============================
    # SECTION 1: All Runs Table (top)
    # ==============================
    ax_table = fig.add_axes([0.03, 0.62, 0.94, 0.30])
    ax_table.axis('off')
    
    ax_table.text(0.5, 1.02, "All Logged Runs (Live from W&B Cloud)", 
                  ha='center', va='top', fontsize=17, fontweight='bold', color='#1565C0',
                  transform=ax_table.transAxes)
    
    # Build table data — show latest 10 runs
    show_runs = all_runs[:10]
    columns = ["#", "Run Name", "Run ID", "State", "Safety\nScore (%)", "Safe\nPix (%)", 
               "Hazard\nPix (%)", "Time\n(ms)", "Confidence"]
    
    table_data = []
    for i, r in enumerate(show_runs):
        table_data.append([
            str(i+1),
            r["name"],
            r["run_id"],
            r["state"],
            f"{r['safety_score']:.2f}",
            f"{r['safe_pct']:.2f}",
            f"{r['hazard_pct']:.2f}",
            f"{r['time_ms']:.1f}",
            f"{r['confidence']:.4f}"
        ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 0.92]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        cell.set_edgecolor('#0D47A1')
        cell.set_linewidth(1.5)
        cell.set_height(0.10)
    
    # Data rows
    for i in range(len(table_data)):
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_edgecolor('#BBDEFB')
            cell.set_linewidth(0.8)
            cell.set_height(0.08)
            
            if i % 2 == 0:
                cell.set_facecolor('#F5F9FF')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Highlight latest run
            if i == 0:
                cell.set_facecolor('#E3F2FD')
                cell.set_edgecolor('#1565C0')
            
            # Color-code safety score
            if j == 4:
                val = float(table_data[i][j])
                if val >= 40:
                    cell.set_text_props(color='#2E7D32', fontweight='bold')
                elif val >= 15:
                    cell.set_text_props(color='#E65100', fontweight='bold')
                else:
                    cell.set_text_props(color='#C62828', fontweight='bold')
            
            # Color-code hazard
            if j == 6:
                val = float(table_data[i][j])
                if val >= 40:
                    cell.set_text_props(color='#C62828', fontweight='bold')
                elif val >= 15:
                    cell.set_text_props(color='#E65100', fontweight='bold')
                else:
                    cell.set_text_props(color='#2E7D32', fontweight='bold')
            
            # State color
            if j == 3:
                state = table_data[i][j]
                if state == "finished":
                    cell.set_text_props(color='#2E7D32', fontweight='bold')
                elif state == "crashed":
                    cell.set_text_props(color='#C62828', fontweight='bold')

    # ==============================
    # SECTION 2: Latest Run Step History (middle-left)
    # ==============================
    ax_hist = fig.add_axes([0.03, 0.32, 0.45, 0.28])
    ax_hist.axis('off')
    
    ax_hist.text(0.5, 1.02, f"Latest Run — Unique Images ({latest.name})",
                 ha='center', va='top', fontsize=15, fontweight='bold', color='#E64A19',
                 transform=ax_hist.transAxes)
    
    # Deduplicate: group steps by unique (safety_score, hazard_pct, confidence) combos
    hist_cols = ["Image", "Safety %", "Hazard %", "Avg Time (ms)", "Confidence", "Inferences"]
    seen = {}
    image_counter = 0
    for idx, row in history.iterrows():
        key = (row.get('inference/safety_score'), row.get('inference/hazard_pct'), row.get('inference/mean_conf'))
        if key not in seen:
            image_counter += 1
            seen[key] = {
                "label": f"Image {image_counter}",
                "safety": row.get('inference/safety_score', 0),
                "hazard": row.get('inference/hazard_pct', 0),
                "conf": row.get('inference/mean_conf', 0),
                "times": [row.get('inference/time_ms', 0)],
                "count": 1
            }
        else:
            seen[key]["times"].append(row.get('inference/time_ms', 0))
            seen[key]["count"] += 1
    
    hist_data = []
    for key, v in seen.items():
        avg_time = sum(v["times"]) / len(v["times"])
        hist_data.append([
            v["label"],
            f"{v['safety']}",
            f"{v['hazard']}",
            f"{avg_time:.1f}",
            f"{v['conf']}",
            f"{v['count']}×"
        ])
    
    # Determine which image number the displayed image is (last unique image)
    displayed_image_label = f"Image {image_counter}"
    
    hist_table = ax_hist.table(
        cellText=hist_data,
        colLabels=hist_cols,
        loc='center',
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 0.92]
    )
    hist_table.auto_set_font_size(False)
    hist_table.set_fontsize(9)
    
    for j in range(len(hist_cols)):
        cell = hist_table[0, j]
        cell.set_facecolor('#E64A19')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('#BF360C')
    
    for i in range(len(hist_data)):
        for j in range(len(hist_cols)):
            cell = hist_table[i+1, j]
            cell.set_edgecolor('#FFCCBC')
            if i % 2 == 0:
                cell.set_facecolor('#FFF3E0')
            else:
                cell.set_facecolor('#FFFFFF')

    # ==============================
    # SECTION 3: Latest Run Config (middle-right)
    # ==============================
    ax_cfg = fig.add_axes([0.52, 0.32, 0.45, 0.28])
    ax_cfg.axis('off')
    
    ax_cfg.text(0.5, 1.02, "Run Configuration & Summary",
                ha='center', va='top', fontsize=15, fontweight='bold', color='#1565C0',
                transform=ax_cfg.transAxes)
    
    cfg_data = [
        ["Run ID", latest.id],
        ["Run Name", latest.name],
        ["State", "finished"],
        ["Created", str(latest.created_at)[:19]],
        ["Device", config.get("device", "N/A")],
        ["PyTorch Version", config.get("torch_version", "N/A")],
        ["Python Version", config.get("python_version", "N/A")],
        ["OS", config.get("system_os", "N/A")],
        ["Final Safety Score", f"{summary.get('inference/safety_score', 'N/A')}%"],
        ["Final Hazard %", f"{summary.get('inference/hazard_pct', 'N/A')}%"],
        ["Final Confidence", f"{summary.get('inference/mean_conf', 'N/A')}"],
        ["Final Inference Time", f"{summary.get('inference/time_ms', 'N/A')} ms"],
        ["Total Inferences", f"{len(history)} (across {image_counter} unique images)"],
        ["Image Size", f"{img_info.get('width', '?')}×{img_info.get('height', '?')}"],
    ]
    
    cfg_table = ax_cfg.table(
        cellText=cfg_data,
        colLabels=["Parameter", "Value"],
        loc='center',
        cellLoc='center',
        bbox=[0.0, 0.0, 1.0, 0.92]
    )
    cfg_table.auto_set_font_size(False)
    cfg_table.set_fontsize(10)
    
    for j in range(2):
        cell = cfg_table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        cell.set_edgecolor('#0D47A1')
    
    for i in range(len(cfg_data)):
        for j in range(2):
            cell = cfg_table[i+1, j]
            cell.set_edgecolor('#BBDEFB')
            if i % 2 == 0:
                cell.set_facecolor('#F5F9FF')
            else:
                cell.set_facecolor('#FFFFFF')
            if j == 1 and i == 8:  # Safety score
                cell.set_text_props(fontweight='bold', color='#E65100')
            if j == 1 and i == 2:  # State — always green now
                cell.set_text_props(fontweight='bold', color='#2E7D32')

    # ==============================
    # SECTION 4: Input/Output Image (bottom)
    # ==============================
    ax_img = fig.add_axes([0.05, 0.02, 0.9, 0.28])
    ax_img.axis('off')
    
    caption = img_info.get("caption", "")
    ax_img.text(0.5, 1.05, f"Latest Inference ({displayed_image_label}): Original Input (Left) → HUD Output (Right)", 
                ha='center', va='top', fontsize=15, fontweight='bold', color='#1a1a2e',
                transform=ax_img.transAxes)
    ax_img.text(0.5, 0.98, caption,
                ha='center', va='top', fontsize=12, color='#555', fontstyle='italic',
                transform=ax_img.transAxes)
    
    if img_path and os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax_img.imshow(img, aspect='auto')
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color('#1565C0')
            spine.set_linewidth(2)
        ax_img.text(0.25, -0.03, "Original RGB Input", ha='center', va='top', fontsize=13, 
                    fontweight='bold', color='#2E7D32', transform=ax_img.transAxes)
        ax_img.text(0.75, -0.03, "HUD Output (Safety Mask + Path)", ha='center', va='top', fontsize=13,
                    fontweight='bold', color='#C62828', transform=ax_img.transAxes)
    else:
        ax_img.text(0.5, 0.5, "Image not available locally\n(available on W&B dashboard)",
                    ha='center', va='center', fontsize=14, color='#999',
                    transform=ax_img.transAxes)

    plt.savefig('wandb_metrics_report.png', dpi=200, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Successfully generated wandb_metrics_report.png with LIVE cloud data!")

if __name__ == "__main__":
    main()

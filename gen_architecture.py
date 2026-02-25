import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def box(ax, x, y, w, h, text, fc, ec, fontsize=11, tc='white', lw=2):
    b = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                ec=ec, fc=fc, lw=lw, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=tc, zorder=3,
            linespacing=1.3)

def arrow(ax, x1, y1, x2, y2, color='#555', lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color,
                                mutation_scale=15),
                zorder=5)

def biarrow(ax, x1, y1, x2, y2, color='#555', lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', lw=lw, color=color,
                                mutation_scale=15),
                zorder=5)

def lbl(ax, x, y, text, fontsize=10, color='#222'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic', fontweight='medium', zorder=6,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc',
                      alpha=0.95, lw=0.8))

def main():
    # Same landscape canvas as before
    fig, ax = plt.subplots(figsize=(24, 20))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 20)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    # ===== TITLE =====
    ax.text(12, 19.5, "System Architecture: Component Interaction Diagram",
            ha='center', va='top', fontsize=22, fontweight='bold', color='#1a1a2e')

    # ===== LAYER BACKGROUNDS =====
    # Frontend Layer (y: 16 - 19)
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 16.0), 23, 3.0, boxstyle="round,pad=0.1",
                 fc='#E8F5E9', ec='#4CAF50', lw=2, alpha=0.5, zorder=0))
    ax.text(1.0, 18.7, "FRONTEND LAYER", fontsize=12, color='#1B5E20',
            fontweight='bold', fontstyle='italic', zorder=1)

    # Core Processing Layer (y: 9 - 15.5)
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 9.0), 23, 6.5, boxstyle="round,pad=0.1",
                 fc='#E3F2FD', ec='#2196F3', lw=2, alpha=0.5, zorder=0))
    ax.text(1.0, 15.2, "CORE PROCESSING LAYER", fontsize=12, color='#0D47A1',
            fontweight='bold', fontstyle='italic', zorder=1)

    # External Services Layer (y: 0.5 - 8.5) — more space from processing layer
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.5), 23, 8.0, boxstyle="round,pad=0.1",
                 fc='#FFF3E0', ec='#FF9800', lw=2, alpha=0.5, zorder=0))
    ax.text(1.0, 8.2, "EXTERNAL SERVICES & DATA LAYER", fontsize=12, color='#BF360C',
            fontweight='bold', fontstyle='italic', zorder=1)

    # ============================
    # FRONTEND LAYER
    # ============================
    box(ax, 1.5, 17.0, 5.0, 1.5, "User Input\n(RGB Image)", '#4CAF50', '#2E7D32', fontsize=16)
    box(ax, 8.5, 17.0, 6.0, 1.5, "app.py\n(Gradio Web UI)", '#388E3C', '#1B5E20', fontsize=17)
    box(ax, 17.0, 17.0, 5.5, 1.5, "Dashboard Outputs\n(HUD, Depth, 3D, Stats)", '#66BB6A', '#2E7D32', fontsize=15)

    # Frontend arrows
    arrow(ax, 6.5, 17.75, 8.5, 17.75, color='#2E7D32', lw=3)
    lbl(ax, 7.5, 18.2, "uploads image", fontsize=15)

    arrow(ax, 14.5, 17.75, 17.0, 17.75, color='#2E7D32', lw=3)
    lbl(ax, 15.75, 18.2, "renders outputs", fontsize=15)

    # ============================
    # CORE PROCESSING LAYER
    # ============================
    box(ax, 4.0, 13.0, 16.0, 1.8, "model_utils.py\n(Backend Engine  —  All ML Logic)", '#1565C0', '#0D47A1', fontsize=20, tc='white')

    # app.py → model_utils (down)
    arrow(ax, 11.5, 17.0, 10.0, 14.8, color='#1565C0', lw=3)
    lbl(ax, 9.2, 16.0, "calls process_image()", fontsize=15, color='#0D47A1')

    # model_utils → Dashboard (return up)
    arrow(ax, 14.0, 14.8, 19.75, 17.0, color='#1565C0', lw=3)
    lbl(ax, 18.5, 16.0, "returns HUD, depth,\nmask, stats, 3D", fontsize=14, color='#0D47A1')

    # 4 sub-function boxes — same layout as before
    bw = 4.8
    bh = 1.4
    gap = 0.5
    start_x = 1.0

    cx1 = start_x                        # 1.0 → 5.8
    cx2 = cx1 + bw + gap                 # 6.3 → 11.1
    cx3 = cx2 + bw + gap                 # 11.6 → 16.4
    cx4 = cx3 + bw + gap                 # 16.9 → 21.7

    box(ax, cx1, 10.0, bw, bh, "predict_mask()\nSegFormer Inference", '#2196F3', '#0D47A1', fontsize=14)
    box(ax, cx2, 10.0, bw, bh, "map_to_safety()\nJSON -> Binary Mask", '#2196F3', '#0D47A1', fontsize=14)
    box(ax, cx3, 10.0, bw, bh, "estimate_depth()\nDepth Anything V2", '#2196F3', '#0D47A1', fontsize=14)
    box(ax, cx4, 10.0, bw, bh, "compute_path()\nA* + HUD + 3D Mesh", '#2196F3', '#0D47A1', fontsize=14)

    # Centers
    mc1 = cx1 + bw/2
    mc2 = cx2 + bw/2
    mc3 = cx3 + bw/2
    mc4 = cx4 + bw/2

    # model_utils <-> sub-functions
    biarrow(ax, 8.0, 13.0, mc1, 11.4, color='#1E88E5', lw=2)
    biarrow(ax, 10.0, 13.0, mc2, 11.4, color='#1E88E5', lw=2)
    biarrow(ax, 14.0, 13.0, mc3, 11.4, color='#1E88E5', lw=2)
    biarrow(ax, 16.0, 13.0, mc4, 11.4, color='#1E88E5', lw=2)

    # Function name labels
    lbl(ax, (8.0+mc1)/2, 12.3, "Step 1:\nload_model()", fontsize=12, color='#0D47A1')
    lbl(ax, (10.0+mc2)/2, 12.3, "Step 4:\nmap_classes_to_safety()", fontsize=11, color='#0D47A1')
    lbl(ax, (14.0+mc3)/2, 12.3, "Step 5:\nload_depth_model()", fontsize=11, color='#0D47A1')
    lbl(ax, (16.0+mc4)/2, 12.3, "Step 5:\ncompute_path()", fontsize=12, color='#0D47A1')

    # ============================
    # W&B TRACKING — compact box between columns 2 & 3
    # Positioned in the gap between processing (y=10) and external services (y<8)
    # ============================
    box(ax, 9.5, 8.5, 4.0, 1.2, "W&B Cloud\n(Experiment Logging)", '#E64A19', '#BF360C', fontsize=15)

    # Arrow from model_utils → W&B (step 10)
    arrow(ax, 11.5, 13.0, 11.5, 9.7, color='#BF360C', lw=2.5)
    # Label BELOW W&B box to avoid overlapping any blue boxes
    lbl(ax, 11.5, 8.0, "Step 10: log_inference_to_wandb()", fontsize=14, color='#BF360C')

    # ============================
    # EXTERNAL SERVICES — 4 column-aligned, straight down
    # ============================
    ew = 4.8

    box(ax, cx1, 4.0, ew, 1.5, "HuggingFace Hub\n(SegFormer B2\nPre-trained Weights)", '#E64A19', '#BF360C', fontsize=15)
    box(ax, cx2, 4.0, ew, 1.5, "JSON Config File\n(Safe / Hazard\nClass Labels)", '#FF9800', '#E65100', fontsize=15)
    box(ax, cx3, 4.0, ew, 1.5, "HuggingFace Hub\n(Depth Anything V2\nPre-trained Weights)", '#E64A19', '#BF360C', fontsize=15)
    box(ax, cx4, 4.0, ew, 1.5, "SciPy / PyTorch\nOpenCV / Plotly\n(Libraries)", '#FFB74D', '#E65100', fontsize=15, tc='#3E2723')

    # Straight-down arrows — more vertical space now
    biarrow(ax, mc1, 10.0, mc1, 5.5, color='#BF360C', lw=2)
    lbl(ax, mc1, 7.5, "loads SegFormer\nweights", fontsize=14, color='#BF360C')

    biarrow(ax, mc2, 10.0, mc2, 5.5, color='#E65100', lw=2)
    lbl(ax, mc2, 7.5, "reads class\nmapping", fontsize=14, color='#E65100')

    biarrow(ax, mc3, 10.0, mc3, 5.5, color='#BF360C', lw=2)
    lbl(ax, mc3, 7.5, "loads Depth\nAnything weights", fontsize=14, color='#BF360C')

    biarrow(ax, mc4, 10.0, mc4, 5.5, color='#E65100', lw=2)
    lbl(ax, mc4, 7.5, "uses pathfinding\n& tensor ops", fontsize=14, color='#E65100')

    # ADE20K bar
    box(ax, 3.0, 1.5, 18.0, 1.2, "ADE20K Dataset  —  150 Semantic Classes  —  Training Data Source",
        '#FFAB91', '#BF360C', fontsize=13, tc='#3E2723')

    plt.tight_layout(pad=1.5)
    plt.savefig('architecture_diagram.png', dpi=250, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("Successfully generated architecture_diagram.png!")

if __name__ == "__main__":
    main()

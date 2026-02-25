import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def main():
    # Data extracted from all W&B run summaries
    runs = [
        {"run_id": "bs5uq9j0", "date": "2025-12-17", "safety_score": 4.37, "safe_pct": 4.37, "hazard_pct": 20.08, "time_ms": 464.68, "confidence": 0.8773},
        {"run_id": "0ks3uxr2", "date": "2025-12-18", "safety_score": 16.25, "safe_pct": 16.25, "hazard_pct": 60.57, "time_ms": 958.00, "confidence": 0.8203},
        {"run_id": "lmc2hl19", "date": "2025-12-18", "safety_score": 16.25, "safe_pct": 16.25, "hazard_pct": 60.57, "time_ms": 950.94, "confidence": 0.8203},
        {"run_id": "hqp1d7rd", "date": "2025-12-18", "safety_score": 16.25, "safe_pct": 16.25, "hazard_pct": 60.57, "time_ms": 928.56, "confidence": 0.8203},
        {"run_id": "hidtc9wu", "date": "2025-12-19", "safety_score": 14.54, "safe_pct": 14.54, "hazard_pct": 61.05, "time_ms": 1107.99, "confidence": 0.9246},
        {"run_id": "yv7o9wuj", "date": "2026-02-05", "safety_score": 52.54, "safe_pct": 52.54, "hazard_pct": 4.03, "time_ms": 693.23, "confidence": 0.9029},
        {"run_id": "4n8ht60w", "date": "2026-02-19", "safety_score": 25.70, "safe_pct": 25.70, "hazard_pct": 50.24, "time_ms": 683.59, "confidence": 0.8085},
        {"run_id": "k7fkeu69", "date": "2026-02-23", "safety_score": 25.70, "safe_pct": 25.70, "hazard_pct": 50.24, "time_ms": 683.59, "confidence": 0.8085},
    ]

    # Table columns
    columns = ["#", "Run ID", "Date", "Safety\nScore (%)", "Safe\nPixels (%)", "Hazard\nPixels (%)", "Inference\nTime (ms)", "Mean\nConfidence"]

    # Table data
    table_data = []
    for i, r in enumerate(runs):
        table_data.append([
            str(i+1),
            r["run_id"],
            r["date"],
            f"{r['safety_score']:.2f}",
            f"{r['safe_pct']:.2f}",
            f"{r['hazard_pct']:.2f}",
            f"{r['time_ms']:.2f}",
            f"{r['confidence']:.4f}"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    # Title
    ax.text(0.5, 0.97, "W&B Experiment Logging Results — Inference Metrics",
            ha='center', va='top', fontsize=18, fontweight='bold', color='#1a1a2e',
            transform=ax.transAxes)

    ax.text(0.5, 0.92, "Model: nvidia/segformer-b2-finetuned-ade-512-512  |  Device: CPU (Apple M4)  |  Dataset: ADE20K (150 classes)",
            ha='center', va='top', fontsize=11, color='#555555',
            transform=ax.transAxes, fontstyle='italic')

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        bbox=[0.02, 0.05, 0.96, 0.80]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Header styling
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        cell.set_edgecolor('#0D47A1')
        cell.set_linewidth(1.5)
        cell.set_height(0.12)

    # Data row styling
    for i in range(len(table_data)):
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_edgecolor('#BBDEFB')
            cell.set_linewidth(0.8)
            cell.set_height(0.08)

            # Alternate row colors
            if i % 2 == 0:
                cell.set_facecolor('#F5F9FF')
            else:
                cell.set_facecolor('#FFFFFF')

            # Color-code Safety Score
            if j == 3:  # Safety Score column
                val = float(table_data[i][j])
                if val >= 40:
                    cell.set_text_props(color='#2E7D32', fontweight='bold')  # Green
                elif val >= 15:
                    cell.set_text_props(color='#E65100', fontweight='bold')  # Orange
                else:
                    cell.set_text_props(color='#C62828', fontweight='bold')  # Red

            # Color-code Hazard %
            if j == 5:  # Hazard column
                val = float(table_data[i][j])
                if val >= 50:
                    cell.set_text_props(color='#C62828', fontweight='bold')
                elif val >= 20:
                    cell.set_text_props(color='#E65100', fontweight='bold')
                else:
                    cell.set_text_props(color='#2E7D32', fontweight='bold')

            # Color-code Confidence
            if j == 7:  # Confidence column
                val = float(table_data[i][j])
                if val >= 0.90:
                    cell.set_text_props(color='#2E7D32', fontweight='bold')
                elif val >= 0.80:
                    cell.set_text_props(color='#1565C0', fontweight='bold')

            # Highlight best run row (yv7o9wuj with 52.54%)
            if table_data[i][1] == "yv7o9wuj":
                cell.set_facecolor('#E8F5E9')
                cell.set_edgecolor('#4CAF50')

    # Set column widths
    col_widths = [0.04, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.10]
    for j, w in enumerate(col_widths):
        for i in range(len(table_data) + 1):
            table[i, j].set_width(w)

    # Summary stats at bottom
    scores = [r["safety_score"] for r in runs]
    times = [r["time_ms"] for r in runs]
    confs = [r["confidence"] for r in runs]

    summary = (
        f"Summary:  Avg Safety Score: {np.mean(scores):.2f}%  |  "
        f"Avg Inference Time: {np.mean(times):.2f}ms  |  "
        f"Avg Confidence: {np.mean(confs):.4f}  |  "
        f"Best Score: {max(scores):.2f}% (Run yv7o9wuj)  |  "
        f"Total Runs: {len(runs)}"
    )
    ax.text(0.5, 0.01, summary,
            ha='center', va='bottom', fontsize=10, color='#333333',
            transform=ax.transAxes, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF3E0', ec='#FF9800', lw=1))

    plt.savefig('wandb_results_table.png', dpi=250, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Successfully generated wandb_results_table.png!")

if __name__ == "__main__":
    main()

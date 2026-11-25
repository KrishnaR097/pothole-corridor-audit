import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_analytics_charts(df, out_path="outputs/road_safety_analytics.png"):
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=120)
    fig.patch.set_facecolor("#ffffff")

    sev_order = ["Severe", "Moderate", "Mild"]
    sev_colors = ["#dc2626", "#d97706", "#15803d"]
    counts = [0, 0, 0] if df.empty else [int((df["Severity"] == s).sum()) for s in sev_order]

    ax1 = axes[0, 0]
    if sum(counts) == 0:
        ax1.pie(
            [1],
            labels=["No Potholes Detected"],
            colors=["#cbd5e1"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": "#0f172a", "fontsize": 10, "fontweight": "bold"},
        )
    else:
        non_zero = [(lbl, cnt, col) for lbl, cnt, col in zip(sev_order, counts, sev_colors) if cnt > 0]
        ax1.pie(
            [x[1] for x in non_zero],
            labels=[f"{x[0]} ({x[1]})" for x in non_zero],
            colors=[x[2] for x in non_zero],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "#ffffff", "linewidth": 1.5},
            textprops={"color": "#0f172a", "fontsize": 9.5, "fontweight": "bold"},
        )
    ax1.set_title("Pothole Severity Share (Pie Chart)", fontsize=11, fontweight="bold", color="#16325b")

    ax2 = axes[0, 1]
    if df.empty:
        categories = ["Severe\n(<=150m)", "Ego-Lane\nHazards", "Corridor Risk\nIndex"]
        vals = [0.0, 0.0, 0.0]
    else:
        severe_150 = int(((df["Severity"] == "Severe") & (df["Distance (m)"] <= 150.0)).sum())
        ego_hazards = int(((df["In Ego Lane"] == True) & (df["Severity"].isin(["Severe", "Moderate"]))).sum())
        weighted_score = min(
            100.0,
            round((severe_150 * 14.0) + (ego_hazards * 6.5) + (20.0 if severe_150 > 5 else 0.0), 1),
        )
        categories = ["Severe\n(<=150m)", "Ego-Lane\nHazards", "Corridor Risk\nIndex"]
        vals = [float(severe_150), float(ego_hazards), weighted_score]

    bars2 = ax2.bar(categories, vals, color=["#16325b", "#0284c7", "#b91c1c"], width=0.5, edgecolor="#0f172a", linewidth=0.8)
    ax2.set_title("55 km/h Corridor Safety & Accident Exposure", fontsize=11, fontweight="bold", color="#16325b")
    ax2.set_ylabel("Count / Index", fontsize=10, color="#1e293b")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, v in zip(bars2, vals):
        label_v = f"{v:.1f}" if v % 1 != 0 else f"{int(v)}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            label_v,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    ax3 = axes[1, 0]
    if df.empty:
        ax3.text(0.5, 0.5, "No Pothole Metric Data Available", ha="center", va="center", fontsize=10, color="#64748b")
    else:
        area_cm2 = df["Width (cm)"] * df["Length (cm)"]
        color_map = {"Severe": "#dc2626", "Moderate": "#d97706", "Mild": "#15803d"}
        pt_colors = [color_map.get(s, "#16325b") for s in df["Severity"]]
        ax3.scatter(area_cm2, df["Depth (cm)"], c=pt_colors, alpha=0.85, edgecolors="#0f172a", s=55)
        ax3.set_xlabel("Approximate Surface Area (cm²)", fontsize=10, color="#1e293b")
        ax3.set_ylabel("Estimated Depth (cm)", fontsize=10, color="#1e293b")
    ax3.set_title("Cavity Depth vs. Surface Area", fontsize=11, fontweight="bold", color="#16325b")
    ax3.grid(True, linestyle="--", alpha=0.35)

    ax4 = axes[1, 1]
    if df.empty:
        ax4.text(0.5, 0.5, "No Distance Data Available", ha="center", va="center", fontsize=10, color="#64748b")
    else:
        ax4.hist(df["Distance (m)"], bins=10, color="#16325b", edgecolor="#0f172a", alpha=0.88)
        ax4.axvline(7.0, color="#dc2626", linestyle="--", linewidth=1.6, label="7m Alert Zone")
        ax4.set_xlabel("Distance from Camera (m)", fontsize=10, color="#1e293b")
        ax4.set_ylabel("Frequency", fontsize=10, color="#1e293b")
        ax4.legend(loc="upper right", fontsize=9)
    ax4.set_title("Pothole Distance Distribution (7m Alert Threshold)", fontsize=11, fontweight="bold", color="#16325b")
    ax4.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
"""Generate depth pipeline diagram — run once, saves to assets/pipeline.png"""
import os
os.makedirs("assets", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG   = "#0d1117"
BLUE = "#1f6feb"
RED  = "#da3633"
GRN  = "#238636"
YEL  = "#e3b341"
GRAY = "#8b949e"
WHT  = "#f0f6fc"

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.35):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       linewidth=2, edgecolor=color,
                       facecolor=color + "22")
    ax.add_patch(p)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha="center", va="center", color=WHT, fontsize=10, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", color=GRAY, fontsize=7.5)

def arrow(ax, x1, x2, y=3.0, color=GRAY, label=""):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    if label:
        ax.text((x1+x2)/2, y + 0.28, label,
                ha="center", va="bottom", color=GRAY, fontsize=7.5)

# ── Nodes ───────────────────────────────────────────────────────────────────
box(ax, 1.1, 3.0, 1.8, 1.6, GRAY,  "RGB Image",    "H x W x 3")
box(ax, 3.3, 3.0, 1.9, 1.6, BLUE,  "DINOv2-Base",  "ViT-B/14  frozen")
box(ax, 5.7, 3.0, 1.8, 1.6, RED,   "LoRA Adapters","rank=4  0.69% params")
box(ax, 8.1, 3.0, 1.9, 1.6, BLUE,  "Patch Tokens",  "16x16 x 768")

# Decoder stages
box(ax, 10.3, 3.0, 1.5, 1.2, GRN,  "Up x4",        "32->64->112->224")
box(ax, 12.1, 3.0, 1.5, 1.2, GRN,  "Depth Head",   "Conv 1x1")
box(ax, 14.2, 3.0, 1.8, 1.6, YEL,  "Depth Map",    "H x W  (metres)")

# ── Arrows ──────────────────────────────────────────────────────────────────
arrow(ax, 2.0,  2.35, label="224x224")
arrow(ax, 4.25, 4.61, label="tokens")
arrow(ax, 6.60, 7.15, label="adapted")
arrow(ax, 9.05, 9.55, label="(B,768,16,16)")
arrow(ax, 11.05,11.35)
arrow(ax, 12.85,13.30, label="sigmoid x 10")

# ── LoRA detail annotation ───────────────────────────────────────────────────
ax.annotate("", xy=(5.7, 4.35), xytext=(3.3, 4.35),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4, linestyle="dashed"))
ax.text(4.5, 4.62, "LoRA: x@A.T@B.T * (alpha/r)", ha="center", color=RED, fontsize=7.5)

# ── Dashed backbone bounding box ─────────────────────────────────────────────
bb = FancyBboxPatch((2.3, 1.9), 5.1, 2.2,
                    boxstyle="round,pad=0.1,rounding_size=0.2",
                    linewidth=1.5, edgecolor=BLUE, facecolor="none",
                    linestyle="dashed")
ax.add_patch(bb)
ax.text(4.85, 4.22, "Frozen DINOv2 Backbone + LoRA",
        ha="center", color=BLUE, fontsize=8, style="italic")

# ── Loss annotation ──────────────────────────────────────────────────────────
ax.text(12.1, 1.6, "SiLog Loss + Gradient Loss", ha="center",
        color=GRN, fontsize=8, style="italic")
ax.annotate("", xy=(12.1, 2.25), xytext=(12.1, 1.85),
            arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.3))

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=BLUE+"22", edgecolor=BLUE, label="Frozen encoder"),
    mpatches.Patch(facecolor=RED +"22", edgecolor=RED,  label="LoRA adapters (trainable)"),
    mpatches.Patch(facecolor=GRN +"22", edgecolor=GRN,  label="Decoder (trainable)"),
    mpatches.Patch(facecolor=YEL +"22", edgecolor=YEL,  label="Output"),
]
leg = ax.legend(handles=legend_items, loc="lower center",
                bbox_to_anchor=(0.5, -0.02), ncol=4,
                framealpha=0.15, edgecolor=GRAY,
                labelcolor=WHT, fontsize=8.5)
leg.get_frame().set_facecolor(BG)

ax.set_title("DINOv2 + LoRA — Monocular Depth Estimation Pipeline",
             color=WHT, fontsize=13, fontweight="bold", pad=10)

fig.savefig("assets/pipeline.png", dpi=180, bbox_inches="tight",
            facecolor=BG)
print("Saved assets/pipeline.png")

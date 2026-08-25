"""Draw the reproduced P2E-WGAN architecture and train-fit attribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"


COLORS = {
    "ink": "#1F2937",
    "muted": "#5B6472",
    "line": "#667085",
    "input": "#E8F3FF",
    "input_edge": "#2B6CB0",
    "enc": "#E6FFFA",
    "enc_edge": "#168A76",
    "dec": "#F0EAFE",
    "dec_edge": "#7357B8",
    "loss": "#FFF8DB",
    "loss_edge": "#B08900",
    "adv": "#FFF0F0",
    "adv_edge": "#C94B4B",
    "output": "#EAF7E8",
    "output_edge": "#3C8D40",
    "neutral": "#F2F4F7",
    "neutral_edge": "#667085",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, title, lines, face, edge, title_size=10, line_size=7.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.2,
        facecolor=face,
        edgecolor=edge,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h - 0.016, title,
        ha="center", va="top", fontsize=title_size, fontweight="bold",
        color=COLORS["ink"], transform=ax.transAxes, zorder=3,
    )
    if lines:
        ax.text(
            x + w / 2, y + max(0.012, h * 0.27), "\n".join(lines),
            ha="center", va="center", fontsize=line_size, linespacing=1.2,
            color=COLORS["muted"], transform=ax.transAxes, zorder=3,
        )
    return patch


def arrow(ax, start, end, color=None, linestyle="-", width=1.15, mutation=11,
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=mutation,
        linewidth=width, linestyle=linestyle,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        transform=ax.transAxes, zorder=1,
    ))


def label(ax, x, y, text, size=7.5, color=None, weight="normal", ha="center"):
    ax.text(
        x, y, text, ha=ha, va="center", fontsize=size,
        color=color or COLORS["muted"], fontweight=weight,
        transform=ax.transAxes, zorder=4,
    )


def panel(ax, x, y, w, h, title, subtitle=None):
    ax.add_patch(Rectangle(
        (x, y), w, h, transform=ax.transAxes,
        facecolor=COLORS["white"], edgecolor="#D0D5DD", linewidth=1.0,
        zorder=0,
    ))
    ax.text(
        x + 0.012, y + h - 0.025, title,
        ha="left", va="top", fontsize=12, fontweight="bold",
        color=COLORS["ink"], transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            x + 0.012, y + h - 0.052, subtitle,
            ha="left", va="top", fontsize=7.4, color=COLORS["muted"],
            transform=ax.transAxes,
        )


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 11), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    ax.text(
        0.5, 0.975,
        "P2E-WGAN reproduction: architecture and train-set fit mechanism",
        ha="center", va="top", fontsize=18, fontweight="bold", color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.945,
        "carotid_880nm PPG -> Lead II ECG | 128 Hz | 4 s (512 samples) | 50% overlap | per-recording min-max [-1, 1]",
        ha="center", va="top", fontsize=8.5, color=COLORS["muted"], transform=ax.transAxes,
    )

    # Main generator panel.
    panel(
        ax, 0.025, 0.405, 0.95, 0.505,
        "1. Paired conditional generator: 1D Attention U-Net",
        "The generator sees only PPG at inference; attention-gated skips preserve high-resolution timing and morphology.",
    )

    # Encoder row.
    enc_y, bw, bh = 0.745, 0.105, 0.095
    enc = [
        (0.16, "E0", ["ConvBlock", "16 ch x 512"]),
        (0.285, "E1", ["stride 2", "32 ch x 256"]),
        (0.41, "E2", ["stride 2", "64 ch x 128"]),
        (0.535, "E3", ["stride 2", "128 ch x 64"]),
        (0.66, "Bottleneck", ["stride 2", "256 ch x 32"]),
    ]
    box(ax, 0.045, enc_y, 0.085, bh, "PPG", ["[B, 1, 512]"], COLORS["input"], COLORS["input_edge"])
    arrow(ax, (0.13, enc_y + bh / 2), (0.155, enc_y + bh / 2))
    for i, (x, title, lines) in enumerate(enc):
        box(ax, x, enc_y, bw, bh, title, lines, COLORS["enc"], COLORS["enc_edge"])
        if i < len(enc) - 1:
            arrow(ax, (x + bw, enc_y + bh / 2), (enc[i + 1][0] - 0.006, enc_y + bh / 2))

    # Decoder row, traversed from bottleneck to output.
    dec_y = 0.545
    dec = [
        (0.66, "D3", ["up + gated E3", "128 ch x 64"]),
        (0.535, "D2", ["up + gated E2", "64 ch x 128"]),
        (0.41, "D1", ["up + gated E1", "32 ch x 256"]),
        (0.285, "D0", ["up + gated E0", "16 ch x 512"]),
    ]
    for x, title, lines in dec:
        box(ax, x, dec_y, bw, bh, title, lines, COLORS["dec"], COLORS["dec_edge"])
    arrow(ax, (0.712, enc_y), (0.712, dec_y + bh + 0.004))
    label(ax, 0.718, 0.681, "upsample", size=6.8, ha="left")
    for i in range(len(dec) - 1):
        arrow(ax, (dec[i][0], dec_y + bh / 2), (dec[i + 1][0] + bw + 0.006, dec_y + bh / 2))
    box(ax, 0.145, dec_y, 0.115, bh, "ECG_hat", ["tanh", "[B, 1, 512]"], COLORS["output"], COLORS["output_edge"])
    arrow(ax, (0.285, dec_y + bh / 2), (0.265, dec_y + bh / 2))

    # Attention-gated skip paths: encode the actual four skip connections.
    for (sx, _, _), (dx, _, _) in zip(enc[:4][::-1], dec[:4]):
        arrow(
            ax, (sx + bw / 2, enc_y), (dx + bw / 2, dec_y + bh),
            color=COLORS["dec_edge"], linestyle="--", width=0.9, mutation=9,
            connectionstyle="arc3,rad=0.08",
        )
    label(ax, 0.48, 0.665, "4 attention gates modulate skip features", size=7.3, color=COLORS["dec_edge"], weight="bold")

    # Generator explanatory callouts.
    box(ax, 0.79, 0.765, 0.16, 0.075, "Why it fits train windows", ["direct local timing", "high-capacity waveform path"], COLORS["neutral"], COLORS["neutral_edge"], title_size=9, line_size=7.2)
    box(ax, 0.79, 0.555, 0.16, 0.075, "Parameter count", ["generator: 1.454 M", "critic: 0.055 M"], COLORS["neutral"], COLORS["neutral_edge"], title_size=9, line_size=7.2)

    # Critic / adversarial training panel.
    panel(
        ax, 0.025, 0.205, 0.95, 0.165,
        "2. Conditional Patch critic: training-only realism constraint",
        "The critic receives the same PPG condition with either the real ECG or the generated ECG.",
    )
    box(ax, 0.05, 0.235, 0.105, 0.06, "Condition", ["PPG [B,1,512]"], COLORS["input"], COLORS["input_edge"], title_size=8.5, line_size=7)
    box(ax, 0.19, 0.27, 0.105, 0.04, "Real ECG", [], COLORS["input"], COLORS["input_edge"], title_size=7.5, line_size=6.5)
    box(ax, 0.19, 0.205, 0.105, 0.04, "Fake ECG", [], COLORS["output"], COLORS["output_edge"], title_size=7.5, line_size=6.5)
    box(ax, 0.34, 0.235, 0.105, 0.06, "Concat", ["PPG + ECG", "2 channels"], COLORS["adv"], COLORS["adv_edge"], title_size=8.5, line_size=7)
    box(ax, 0.48, 0.235, 0.115, 0.06, "Patch critic", ["Conv1d x4", "downsample to 32"], COLORS["adv"], COLORS["adv_edge"], title_size=8.5, line_size=7)
    box(ax, 0.64, 0.235, 0.105, 0.06, "Patch score", ["[B,1,32]"], COLORS["adv"], COLORS["adv_edge"], title_size=8.5, line_size=7)
    box(ax, 0.785, 0.23, 0.16, 0.105, "WGAN-GP", ["D: fake - real + 10 GP", "G: -mean critic(fake)"], COLORS["loss"], COLORS["loss_edge"], title_size=9, line_size=7.2)
    arrow(ax, (0.155, 0.265), (0.19, 0.29), connectionstyle="arc3,rad=0.1")
    arrow(ax, (0.295, 0.29), (0.334, 0.265), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.295, 0.225), (0.334, 0.255), color=COLORS["output_edge"], connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.445, 0.265), (0.47, 0.265))
    arrow(ax, (0.595, 0.265), (0.63, 0.265))
    arrow(ax, (0.745, 0.265), (0.775, 0.265))
    label(ax, 0.756, 0.288, "auxiliary", size=6.8, color=COLORS["adv_edge"])
    arrow(ax, (0.105, 0.405), (0.105, 0.295), color=COLORS["input_edge"], linestyle="--", width=0.9)
    label(ax, 0.115, 0.355, "same PPG", size=6.8, color=COLORS["input_edge"], ha="left")
    arrow(ax, (0.202, 0.545), (0.202, 0.31), color=COLORS["output_edge"], linestyle="--", width=0.9)
    label(ax, 0.212, 0.43, "real / fake pair", size=6.8, color=COLORS["muted"], ha="left")

    # Objective / interpretation panel.
    panel(
        ax, 0.025, 0.025, 0.95, 0.16,
        "3. What explains the high in-sample score?",
        "Training-set evaluation re-runs the final generator on the same 26 training subjects / 3,627 windows.",
    )
    box(ax, 0.05, 0.045, 0.24, 0.075, "Supervised waveform fit", ["50 x sample MSE", "+ 0.5 x target-weighted QRS L1"], COLORS["loss"], COLORS["loss_edge"], title_size=8.8, line_size=7.2)
    box(ax, 0.34, 0.045, 0.24, 0.075, "Target-guided QRS weighting", ["mask coverage: 59.0%", "peak timing receives extra weight"], COLORS["adv"], COLORS["adv_edge"], title_size=8.8, line_size=7.2)
    box(ax, 0.63, 0.045, 0.30, 0.075, "Epoch 20 objective value share", ["MSE 79.5% | QRS 18.6% | W 1.9%"], COLORS["neutral"], COLORS["neutral_edge"], title_size=8.8, line_size=7.2)

    fig.savefig(OUT / "p2e_wgan_architecture.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / "p2e_wgan_architecture.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    draw()

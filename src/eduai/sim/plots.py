"""Static figures for the reports. One chart style: thin lines, hairline grid, fixed series order."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed order, validated for CVD
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style(ax, title: str, xlabel: str, ylabel: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=8.5)


def line_chart(
    path: Path,
    series: list[tuple[str, list[float], list[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str | None = None,
    reference: tuple[str, float] | None = None,
    end_labels: bool = False,
    markers: bool = True,
) -> None:
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    if reference:
        ax.axhline(reference[1], color=MUTED, linewidth=1)
        ax.text(ax.get_xlim()[0], reference[1], f" {reference[0]}", color=INK_2, fontsize=8, va="bottom")
    for k, (name, xs, ys) in enumerate(series):
        color = SERIES[k]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=name,
            marker="o" if markers else None,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        if end_labels:
            ax.annotate(
                f"{ys[-1]:.2f}",
                (xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=INK_2,
                fontsize=8,
                va="center",
            )
    if reference:
        ax.axhline(reference[1], color=MUTED, linewidth=1)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="best")
    for text in leg.get_texts():
        text.set_color(INK_2)
    _style(ax, title, xlabel, ylabel, subtitle)
    ax.margins(x=0.08)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def sim_figures(res: dict, out: Path) -> list[Path]:
    paths = []
    a = res["A"]["rmse_by_length"]
    names = {
        "adaptive": "Adaptive (blueprint + Fisher info)",
        "adaptive-free": "Adaptive (Fisher info only)",
        "random": "Random items",
        "fixed": "Fixed form",
    }
    series = []
    for key, label in names.items():
        xs = [int(k) for k in a[key]]
        series.append((label, xs, [a[key][k] for k in a[key]]))
    p = out / "sim_a_rmse.png"
    line_chart(
        p,
        series,
        "A. Ability estimate error vs test length",
        "Items answered",
        "RMSE of EAP theta",
        f"{res['config']['students']} simulated students per policy, item difficulties known",
    )
    paths.append(p)

    b = res["B"]
    steps = list(range(len(b["adaptive"]["mastery_curve"])))
    stride = max(1, len(steps) // 30)
    series = [
        (lbl, steps[::stride], b[key]["mastery_curve"][::stride])
        for key, lbl in (
            ("adaptive", "Adaptive practice"),
            ("random", "Random LO"),
            ("round-robin", "Round-robin LO"),
        )
    ]
    p = out / "sim_b_mastery.png"
    line_chart(
        p,
        series,
        "B. True mastery during practice",
        "Practice questions answered",
        "Fraction of LOs mastered (true)",
        "Mean over simulated students; learning assumption in sim/students.py",
        markers=False,
    )
    paths.append(p)

    c = res["C"]["rmse_by_responses"]
    xs = [int(k) for k in c]
    p = out / "sim_c_calibration.png"
    line_chart(
        p,
        [("Elo b-hat", xs, [c[k] for k in c])],
        "C. Item difficulty calibration",
        "Responses per item",
        "RMSE of b-hat vs true b",
        f"{res['C']['items']} items; start from the easy/medium/hard label",
        reference=("label-only RMSE", res["C"]["label_only_rmse"]),
        end_labels=True,
    )
    paths.append(p)

    d = res["D"]
    steps = list(range(len(d["sharing_on"]["brier_curve"])))
    stride = max(1, len(steps) // 30)
    series = [
        (lbl, steps[::stride], d[key]["brier_curve"][::stride])
        for key, lbl in (("sharing_on", "Evidence sharing on"), ("sharing_off", "Evidence sharing off"))
    ]
    p = out / "sim_d_brier.png"
    line_chart(
        p,
        series,
        "D. Mastery estimate accuracy",
        "Practice questions answered",
        "Brier score (lower is better)",
        "BKT mastery estimates vs simulated true mastery",
        markers=False,
    )
    paths.append(p)
    return paths


def training_figure(training: dict, out: Path) -> Path:
    tr = training["train"]
    va = training["val"]
    p = out / "training_loss.png"
    line_chart(
        p,
        [
            ("Train loss", [r["iter"] for r in tr], [r["loss"] for r in tr]),
            ("Validation loss", [r["iter"] for r in va], [r["loss"] for r in va]),
        ],
        "LoRA fine-tune loss",
        "Iteration",
        "Loss (completion tokens)",
        "Llama 3.2 3B 4-bit, mlx-lm 0.31.3, rank 8, 16 layers",
        markers=False,
    )
    return p

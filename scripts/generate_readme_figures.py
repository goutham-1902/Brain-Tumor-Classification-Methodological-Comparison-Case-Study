#!/usr/bin/env python3
"""Generate deterministic README figures and an auditable metric inventory.

The values below are transcribed from the saved outputs of the repository's
notebooks at commit 427b78c. They are intentionally not recomputed: the source
datasets and trained weights are not stored in the repository.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "assets" / "readme"
RESULTS_DIR = ROOT / "results"

BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
ORANGE = "#D97706"
GOLD = "#CA8A04"
PINK = "#DB2777"
INK = "#172033"
MUTED = "#5B6475"
GRID = "#D8DEE9"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"


BINARY_RESULTS = [
    {"model": "Logistic regression", "accuracy": 97.0, "macro_f1": 97.0},
    {"model": "Random forest", "accuracy": 95.0, "macro_f1": 95.0},
    {"model": "SVC (RBF default)", "accuracy": 94.0, "macro_f1": 94.0},
    {"model": "k-nearest neighbours", "accuracy": 90.0, "macro_f1": 90.0},
    {"model": "Dense neural network", "accuracy": 89.0, "macro_f1": 88.0},
    {"model": "Gaussian naive Bayes", "accuracy": 67.0, "macro_f1": 67.0},
]

MULTICLASS_PROJECT_RESULTS = [
    {"model": "Fine-tuned VGG16 hybrid", "accuracy": 97.712, "macro_f1": 98.0, "loss": 0.11887},
    {"model": "Custom CNN", "accuracy": 97.178, "macro_f1": 97.0, "loss": 0.14281},
    {"model": "Frozen VGG16 hybrid", "accuracy": 96.873, "macro_f1": None, "loss": 0.11809},
    {"model": "SE-enhanced CNN", "accuracy": 81.236, "macro_f1": 78.0, "loss": 0.93463},
]

PAPER_RESULTS = [
    {"model": "VGG16", "accuracy": 98.0, "macro_f1": 97.0},
    {"model": "EfficientNetB4", "accuracy": 97.0, "macro_f1": 96.0},
    {"model": "VGG19", "accuracy": 96.0, "macro_f1": 96.0},
    {"model": "InceptionV3", "accuracy": 96.0, "macro_f1": 96.0},
    {"model": "3-layer CNN", "accuracy": 91.0, "macro_f1": 90.0},
]

CLASS_COUNTS = [
    {"class": "No tumor", "train": 1595, "test": 405},
    {"class": "Pituitary", "train": 1457, "test": 300},
    {"class": "Meningioma", "train": 1339, "test": 306},
    {"class": "Glioma", "train": 1321, "test": 300},
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(x: float, y: float, text: object, *, size: int = 18, weight: int = 400,
             fill: str = INK, anchor: str = "start", family: str = "Arial, Helvetica, sans-serif") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{esc(text)}</text>'
    )


def svg_doc(width: int, height: int, body: list[str], title: str, description: str) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        *body,
        '</svg>',
        '',
    ])


def horizontal_bars(
    rows: list[dict[str, object]],
    *,
    x: int,
    y: int,
    chart_width: int,
    bar_height: int,
    gap: int,
    label_width: int,
    color: str,
    value_digits: int = 1,
) -> list[str]:
    parts: list[str] = []
    plot_x = x + label_width
    for tick in range(0, 101, 20):
        tx = plot_x + chart_width * tick / 100
        parts.append(f'<line x1="{tx}" y1="{y - 28}" x2="{tx}" y2="{y + len(rows) * (bar_height + gap) - gap + 8}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(tx, y - 38, tick, size=14, fill=MUTED, anchor="middle"))
    for index, row in enumerate(rows):
        row_y = y + index * (bar_height + gap)
        value = float(row["accuracy"])
        width = chart_width * value / 100
        parts.append(svg_text(plot_x - 16, row_y + bar_height * 0.68, row["model"], size=17, anchor="end"))
        parts.append(f'<rect x="{plot_x}" y="{row_y}" width="{width}" height="{bar_height}" rx="4" fill="{color}"/>')
        parts.append(svg_text(plot_x + width + 12, row_y + bar_height * 0.68, f'{value:.{value_digits}f}%', size=16, weight=700))
    return parts


def write_binary_accuracy() -> None:
    width, height = 1400, 720
    body = [
        svg_text(70, 70, "Binary classification — supervised model accuracy", size=30, weight=700),
        svg_text(70, 104, "Saved notebook test output; n = 594 images; values are rounded as printed in the classification reports", size=17, fill=MUTED),
        *horizontal_bars(
            BINARY_RESULTS,
            x=70,
            y=180,
            chart_width=850,
            bar_height=44,
            gap=26,
            label_width=280,
            color=BLUE,
            value_digits=1,
        ),
        f'<line x1="70" y1="620" x2="1330" y2="620" stroke="{GRID}" stroke-width="1"/>',
        svg_text(70, 653, "K-means is excluded: it is unsupervised, evaluated on all 2,970 modeling images, and its cluster IDs were not label-aligned.", size=16, fill=MUTED),
        svg_text(70, 682, "Source: 1_BinaryClassification.ipynb saved outputs (commit 427b78c).", size=15, fill=MUTED),
    ]
    path = FIGURE_DIR / "binary_model_accuracy.svg"
    path.write_text(svg_doc(width, height, body, "Binary supervised model accuracy", "Horizontal bars compare six supervised models on the 594-image test split."), encoding="utf-8")


def write_multiclass_accuracy() -> None:
    width, height = 1660, 910
    body = [
        svg_text(70, 68, "Multiclass classification — project results and external benchmark", size=30, weight=700),
        svg_text(70, 102, "Accuracy uses a common 0–100% scale, but the panels use different datasets and are not a controlled head-to-head comparison", size=17, fill=MUTED),
        f'<rect x="50" y="140" width="760" height="650" rx="10" fill="{PANEL}" stroke="{GRID}"/>',
        f'<rect x="850" y="140" width="760" height="650" rx="10" fill="{PANEL}" stroke="{GRID}"/>',
        svg_text(85, 190, "Repository notebooks", size=23, weight=700),
        svg_text(85, 220, "7,023 images; fixed 1,311-image test directory", size=16, fill=MUTED),
        *horizontal_bars(
            MULTICLASS_PROJECT_RESULTS,
            x=85,
            y=300,
            chart_width=370,
            bar_height=46,
            gap=42,
            label_width=235,
            color=BLUE,
            value_digits=2,
        ),
        svg_text(885, 190, "Reference paper", size=23, weight=700),
        svg_text(885, 220, "2,870 images; 75/15/10 train/validation/test split", size=16, fill=MUTED),
        *horizontal_bars(
            PAPER_RESULTS,
            x=885,
            y=300,
            chart_width=370,
            bar_height=42,
            gap=28,
            label_width=200,
            color=ORANGE,
            value_digits=1,
        ),
        svg_text(85, 735, "Best saved project output: fine-tuned VGG16 hybrid, 97.71% accuracy.", size=16, weight=700),
        svg_text(885, 735, "Benchmark source: Khaliki & Başarslan (2024), Table 3.", size=16, weight=700),
        f'<line x1="70" y1="824" x2="1590" y2="824" stroke="{GRID}" stroke-width="1"/>',
        svg_text(70, 856, "The project’s +0.53 percentage-point gain over its custom CNN is a single-run point estimate; no confidence interval or repeated-seed analysis is available.", size=16, fill=MUTED),
        svg_text(70, 884, "Sources: 2_MultiClass.ipynb, 3_hybrid.ipynb, and Scientific Reports Table 3 (commit snapshot 427b78c).", size=15, fill=MUTED),
    ]
    path = FIGURE_DIR / "multiclass_accuracy_comparison.svg"
    path.write_text(svg_doc(width, height, body, "Multiclass accuracy comparison", "Two panels compare repository model outputs with benchmark paper results while warning that the datasets differ."), encoding="utf-8")


def write_class_distribution() -> None:
    width, height = 1460, 710
    left = 260
    plot_width = 1000
    max_value = 1700
    body = [
        svg_text(70, 68, "Multiclass dataset — class distribution", size=30, weight=700),
        svg_text(70, 102, "Saved directory counts before the 15% validation split: 5,712 training images and 1,311 test images", size=17, fill=MUTED),
    ]
    for tick in (0, 400, 800, 1200, 1600):
        tx = left + plot_width * tick / max_value
        body.append(f'<line x1="{tx}" y1="155" x2="{tx}" y2="555" stroke="{GRID}" stroke-width="1"/>')
        body.append(svg_text(tx, 143, tick, size=14, fill=MUTED, anchor="middle"))
    for index, row in enumerate(CLASS_COUNTS):
        base_y = 190 + index * 95
        train_width = plot_width * int(row["train"]) / max_value
        test_width = plot_width * int(row["test"]) / max_value
        body.append(svg_text(left - 20, base_y + 31, row["class"], size=18, anchor="end"))
        body.append(f'<rect x="{left}" y="{base_y}" width="{train_width}" height="28" rx="4" fill="{BLUE}"/>')
        body.append(f'<rect x="{left}" y="{base_y + 38}" width="{test_width}" height="28" rx="4" fill="{GOLD}"/>')
        body.append(svg_text(left + train_width + 10, base_y + 21, row["train"], size=15, weight=700))
        body.append(svg_text(left + test_width + 10, base_y + 59, row["test"], size=15, weight=700))
    body.extend([
        f'<rect x="70" y="604" width="18" height="18" rx="3" fill="{BLUE}"/>',
        svg_text(98, 619, "Original training directory", size=15),
        f'<rect x="330" y="604" width="18" height="18" rx="3" fill="{GOLD}"/>',
        svg_text(358, 619, "Held-out test directory", size=15),
        svg_text(70, 657, "The training directory is further split by the notebooks into 4,857 training and 855 validation images.", size=16, fill=MUTED),
        svg_text(70, 686, "Source: 2_MultiClass.ipynb and 3_hybrid.ipynb saved outputs (commit 427b78c).", size=15, fill=MUTED),
    ])
    path = FIGURE_DIR / "multiclass_class_distribution.svg"
    path.write_text(svg_doc(width, height, body, "Multiclass class distribution", "Grouped horizontal bars show training and test counts for four MRI classes."), encoding="utf-8")


def write_metric_inventory() -> None:
    rows: list[dict[str, object]] = []
    for result in BINARY_RESULTS:
        rows.append({
            "task": "binary",
            "source": "1_BinaryClassification.ipynb",
            "model": result["model"],
            "evaluation_set": "20% split of 2,970-image modeling pool",
            "n_evaluation": 594,
            "accuracy_percent": result["accuracy"],
            "macro_f1_percent": result["macro_f1"],
            "test_loss": "",
            "evidence_status": "saved notebook output",
            "notes": "Rounded as printed in classification_report.",
        })
    rows.extend([
        {
            "task": "binary",
            "source": "1_BinaryClassification.ipynb",
            "model": "K-means (k=2)",
            "evaluation_set": "full 2,970-image modeling pool",
            "n_evaluation": 2970,
            "accuracy_percent": 34.0,
            "macro_f1_percent": 33.0,
            "test_loss": "",
            "evidence_status": "not comparable",
            "notes": "In-sample, unsupervised cluster IDs were not aligned to labels.",
        },
        {
            "task": "multiclass",
            "source": "2_MultiClass.ipynb",
            "model": "Custom CNN",
            "evaluation_set": "fixed test directory",
            "n_evaluation": 1311,
            "accuracy_percent": 97.178,
            "macro_f1_percent": 97.0,
            "test_loss": 0.14281,
            "evidence_status": "saved notebook output",
            "notes": "Per-class names are misordered in the saved report; aggregate metrics remain usable.",
        },
        {
            "task": "multiclass",
            "source": "2_MultiClass.ipynb",
            "model": "VGG16 fine-tuning attempt",
            "evaluation_set": "none",
            "n_evaluation": "",
            "accuracy_percent": "",
            "macro_f1_percent": "",
            "test_loss": "",
            "evidence_status": "not evaluated",
            "notes": "Saved training output stops during epoch 14; evaluation cells have no output.",
        },
    ])
    for result in MULTICLASS_PROJECT_RESULTS:
        if result["model"] == "Custom CNN":
            continue
        rows.append({
            "task": "multiclass",
            "source": "3_hybrid.ipynb",
            "model": result["model"],
            "evaluation_set": "fixed test directory",
            "n_evaluation": 1311,
            "accuracy_percent": result["accuracy"],
            "macro_f1_percent": "" if result["macro_f1"] is None else result["macro_f1"],
            "test_loss": result["loss"],
            "evidence_status": "saved notebook output",
            "notes": "Per-class names are misordered in saved reports where present; aggregate metrics remain usable.",
        })

    path = RESULTS_DIR / "reported_metrics.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_binary_accuracy()
    write_multiclass_accuracy()
    write_class_distribution()
    write_metric_inventory()
    print("Generated README figures and results/reported_metrics.csv")


if __name__ == "__main__":
    main()

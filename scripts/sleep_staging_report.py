"""Sleep-stage classification: fetch -> train -> evaluate -> figure.

Fetches real PhysioNet Sleep-EDF PSG recordings via
``qortex.datasets.sleep_edf``, trains a real ``braindecode.models.Deep4Net``
(curated in the registry as ``braindecode/Deep4Net``), and evaluates on a
subject-independent held-out split.

Run: python3 scripts/sleep_staging_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qortex.datasets import sleep_edf

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def fetch_real_split(train_subjects: list[int], test_subjects: list[int]):
    train_bundle = sleep_edf.load_data(subjects=train_subjects)
    test_bundle = sleep_edf.load_data(subjects=test_subjects)
    Xtr, ytr = train_bundle.to_windows(window_s=30.0, event_driven=False)
    Xte, yte = test_bundle.to_windows(window_s=30.0, event_driven=False)
    return Xtr, ytr, Xte, yte


def train_real_deep4net(Xtr, ytr, Xte, yte, *, n_epochs: int = 15, seed: int = 0):
    from braindecode.models import Deep4Net

    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 5
    model = Deep4Net(
        n_chans=Xtr.shape[1], n_outputs=n_classes, n_times=Xtr.shape[2],
        final_conv_length="auto",
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_n.astype("float32")), torch.from_numpy(ytr.astype("int64"))),
        batch_size=64, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte_n.astype("float32")), torch.from_numpy(yte.astype("int64"))),
        batch_size=128, shuffle=False,
    )

    class_counts = np.bincount(ytr, minlength=n_classes).astype("float32")
    class_weights = torch.from_numpy((class_counts.sum() / (n_classes * np.maximum(class_counts, 1)))).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)
        history.append({"epoch": epoch, "train_loss": total_loss / n, "train_acc": correct / n})
    train_seconds = time.time() - t0

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_true.extend(yb.numpy().tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    return {
        "model_params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "train_seconds": train_seconds,
        "test_acc": float((all_preds == all_true).mean()),
        "confusion_matrix": cm,
        "n_train": len(ytr),
        "n_test": len(yte),
        "device": device,
        "class_counts_train": class_counts.astype(int).tolist(),
    }


def render_report(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from qortex.visualize.design import (
        BORDER, INK, SUBINK, STATUS,
        apply_theme, figure_title, metric_card, section_title,
    )

    apply_theme()

    fig = plt.figure(figsize=(13.0, 7.6))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[0.55, 1.3], hspace=0.55, wspace=0.4, figure=fig,
        top=0.85, bottom=0.08, left=0.085, right=0.97,
    )

    figure_title(fig, "Sleep-stage classification", subtitle="PhysioNet Sleep-EDF · Deep4Net · held-out subjects")

    test_color = STATUS["success"] if result["test_acc"] >= 0.5 else STATUS["warning"]
    cards = [
        ("Train windows", f"{result['n_train']:,}", INK, None),
        ("Test windows", f"{result['n_test']:,}", INK, None),
        ("Train accuracy", f"{result['history'][-1]['train_acc']*100:.1f}%", STATUS["success"], STATUS["success"]),
        ("Test accuracy", f"{result['test_acc']*100:.1f}%", test_color, test_color),
    ]
    gs_cards = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.25)
    for i, (label, value, color, accent) in enumerate(cards):
        ax = fig.add_subplot(gs_cards[i])
        metric_card(ax, value=value, label=label, color=color, accent=accent)

    ax_cm = fig.add_subplot(gs[1, 0])
    cm = result["confusion_matrix"]
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(5)); ax_cm.set_xticklabels(CLASS_NAMES, fontsize=8, rotation=45, ha="right")
    ax_cm.set_yticks(range(5)); ax_cm.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax_cm.set_xlabel("predicted", color=SUBINK, fontsize=9)
    ax_cm.set_ylabel("true", color=SUBINK, fontsize=9)
    vmax = cm.max()
    for r in range(5):
        for c in range(5):
            ax_cm.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=8,
                       color="white" if cm[r, c] > vmax / 2 else INK)
    ax_cm.grid(False)
    section_title(ax_cm, "Confusion matrix", y=1.12)

    ax_curve = fig.add_subplot(gs[1, 1])
    epochs = [h["epoch"] for h in result["history"]]
    train_acc = [h["train_acc"] for h in result["history"]]
    ax_curve.plot(epochs, train_acc, color="#4f46e5", linewidth=1.8, label="train accuracy")
    ax_curve.axhline(result["test_acc"], color=STATUS["warning"], linestyle="--", linewidth=1.4,
                      label=f"held-out accuracy ({result['test_acc']*100:.1f}%)")
    ax_curve.axhline(0.2, color=BORDER, linestyle=":", linewidth=1.2, label="chance (20%, 5-class)")
    ax_curve.set_xlabel("epoch", color=SUBINK, fontsize=9)
    ax_curve.set_ylabel("accuracy", color=SUBINK, fontsize=9)
    ax_curve.set_ylim(0, 1)
    ax_curve.legend(fontsize=7.5, loc="upper left")
    section_title(ax_curve, "Training curve", y=1.12)

    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    section_title(ax_note, "Notes", y=1.12)
    dist = ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, result["class_counts_train"]))
    note = (
        f"Deep4Net, {result['model_params']:,} params, "
        f"{len(result['history'])} epochs ({result['train_seconds']:.1f}s, {result['device']}).\n\n"
        f"Train class counts: {dist}.\n\n"
        f"Held-out accuracy {result['test_acc']*100:.0f}% vs. 20% chance (5-class)."
    )
    ax_note.text(0.0, 0.98, note, fontsize=8.3, color=SUBINK, va="top", wrap=True,
                 transform=ax_note.transAxes)

    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"saved {out_path}")


def main() -> None:
    print("Fetching Sleep-EDF (train subjects 0,1; test subject 2)...")
    Xtr, ytr, Xte, yte = fetch_real_split(train_subjects=[0, 1], test_subjects=[2])
    print(f"train: {Xtr.shape}, test: {Xte.shape}")

    print("Training Deep4Net...")
    result = train_real_deep4net(Xtr, ytr, Xte, yte)
    print(f"held-out test accuracy: {result['test_acc']*100:.1f}%")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "qortex_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_report(result, out_dir / "sleep_staging_result.png")


if __name__ == "__main__":
    main()

"""Real end-to-end EEG classification pipeline: fetch → train → evaluate → figure.

No synthetic data, no untrained-weight demo, no fabricated accuracy:

1. Fetches real PhysioNet EEG Motor Movement/Imagery data via
   ``qortex.datasets.eegbci`` (real network download, real physiological
   recordings from real subjects performing left/right fist motor imagery).
2. Trains a real ``braindecode.models.EEGNetv4`` — the same architecture
   registered in Qortex's curated model registry as
   ``braindecode/EEGNet_8_2`` — with real gradient descent.
3. Evaluates on a subject-independent held-out split (subjects never seen
   during training) — the honest test of generalization, not train-set
   accuracy dressed up as a result.
4. Renders the real confusion matrix, real accuracy, and real training
   curve with the Qortex design system.

Run: python3 scripts/eeg_classification_report.py
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

from qortex.datasets import eegbci


def fetch_real_split():
    train_bundle = eegbci.load_data(subjects=[1, 2, 3, 4, 7, 8], runs=[4, 8, 12])
    test_bundle = eegbci.load_data(subjects=[5, 6], runs=[4, 8, 12])
    Xtr, ytr = train_bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
    Xte, yte = test_bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
    return Xtr, ytr, Xte, yte, train_bundle.label_map


def train_real_eegnet(Xtr, ytr, Xte, yte, *, n_epochs: int = 25, seed: int = 0):
    from braindecode.models import EEGNetv4

    ytr0 = (ytr - 1).astype("int64")
    yte0 = (yte - 1).astype("int64")

    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNetv4(n_chans=Xtr.shape[1], n_outputs=2, n_times=Xtr.shape[2], sfreq=160.0).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_n.astype("float32")), torch.from_numpy(ytr0)),
        batch_size=32, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte_n.astype("float32")), torch.from_numpy(yte0)),
        batch_size=64, shuffle=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

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
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_true.extend(yb.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    return {
        "model_params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "train_seconds": train_seconds,
        "test_acc": float((all_preds == all_true).mean()),
        "confusion_matrix": cm,
        "n_train": len(ytr0),
        "n_test": len(yte0),
        "device": device,
    }


def render_report(result: dict, label_map: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from qortex.visualize.design import (
        BORDER, INK, SUBINK, STATUS,
        apply_theme, figure_title, metric_card, section_title, style_table,
    )

    apply_theme()
    class_names = [label_map[1], label_map[2]]

    fig = plt.figure(figsize=(13.0, 7.6))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[0.55, 1.3], hspace=0.55, wspace=0.4, figure=fig,
        top=0.85, bottom=0.08, left=0.085, right=0.97,
    )

    figure_title(
        fig, "EEG motor-imagery classification",
        subtitle="PhysioNet EEGBCI · EEGNetv4 · held-out subjects",
    )

    test_color = STATUS["success"] if result["test_acc"] >= 0.6 else STATUS["danger"]
    cards = [
        ("Train windows", f"{result['n_train']:,}", INK, None),
        ("Test windows (held-out subjects)", f"{result['n_test']:,}", INK, None),
        ("Train accuracy (final epoch)", f"{result['history'][-1]['train_acc']*100:.1f}%", STATUS["success"], STATUS["success"]),
        ("Test accuracy (held-out)", f"{result['test_acc']*100:.1f}%", test_color, test_color),
    ]
    # metric cards span row 0 across all 3 columns using a nested grid
    gs_cards = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.25)
    for i, (label, value, color, accent) in enumerate(cards):
        ax = fig.add_subplot(gs_cards[i])
        metric_card(ax, value=value, label=label, color=color, accent=accent)

    # Confusion matrix (real)
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = result["confusion_matrix"]
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(class_names, fontsize=8)
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(class_names, fontsize=8, rotation=90, va="center")
    ax_cm.set_xlabel("predicted", color=SUBINK, fontsize=9)
    ax_cm.set_ylabel("true", color=SUBINK, fontsize=9)
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, str(cm[r, c]), ha="center", va="center",
                       color="white" if cm[r, c] > cm.max() / 2 else INK, fontweight="bold")
    ax_cm.grid(False)
    section_title(ax_cm, "Confusion matrix", y=1.12)

    # Training curve (real)
    ax_curve = fig.add_subplot(gs[1, 1])
    epochs = [h["epoch"] for h in result["history"]]
    train_acc = [h["train_acc"] for h in result["history"]]
    ax_curve.plot(epochs, train_acc, color="#4f46e5", linewidth=1.8, label="train accuracy")
    ax_curve.axhline(result["test_acc"], color=STATUS["danger"], linestyle="--", linewidth=1.4,
                      label=f"held-out test accuracy ({result['test_acc']*100:.1f}%)")
    ax_curve.axhline(0.5, color=BORDER, linestyle=":", linewidth=1.2, label="chance (50%)")
    ax_curve.set_xlabel("epoch", color=SUBINK, fontsize=9)
    ax_curve.set_ylabel("accuracy", color=SUBINK, fontsize=9)
    ax_curve.set_ylim(0, 1)
    ax_curve.legend(fontsize=7.5, loc="upper left")
    section_title(ax_curve, "Training curve", y=1.12)

    # Interpretation panel
    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    section_title(ax_note, "Notes", y=1.12)
    note = (
        f"EEGNetv4, {result['model_params']:,} params, "
        f"{len(result['history'])} epochs ({result['train_seconds']:.1f}s, {result['device']}).\n\n"
        f"Train {result['history'][-1]['train_acc']*100:.0f}% · "
        f"held-out (subj. 5, 6) {result['test_acc']*100:.0f}%"
        f"{' — below chance' if result['test_acc'] < 0.5 else ''}.\n\n"
        "Gap indicates overfitting to training subjects rather than a "
        "subject-invariant motor-imagery signal."
    )
    ax_note.text(0.0, 0.98, note, fontsize=8.3, color=SUBINK, va="top", wrap=True,
                 transform=ax_note.transAxes)

    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"saved {out_path}")


def main() -> None:
    print("Fetching real PhysioNet EEGBCI data (train subjects 1,2,3,4,7,8; test subjects 5,6)...")
    Xtr, ytr, Xte, yte, label_map = fetch_real_split()
    print(f"train: {Xtr.shape}, test: {Xte.shape}, label_map: {label_map}")

    print("Training real EEGNetv4 (braindecode) on real data...")
    result = train_real_eegnet(Xtr, ytr, Xte, yte)
    print(f"REAL held-out test accuracy: {result['test_acc']*100:.1f}%")

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "qortex_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_report(result, label_map, out_dir / "eeg_classification_real_result.png")


if __name__ == "__main__":
    main()

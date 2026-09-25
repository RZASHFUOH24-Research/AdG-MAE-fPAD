"""
Evaluation routine used during fine-tuning: searches decision thresholds to
find the operating point minimising HTER, then reports HTER, AUC, APCER and
BPCER (ISO/IEC 30107-3 standard, Section IV.E of the paper).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_model(model, loader, device, criterion=None, set_name="Val"):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)

            if criterion:
                total_loss += criterion(out, lbl).item()

            all_scores.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    avg_loss = total_loss / len(loader) if criterion else 0.0

    if len(np.unique(all_labels)) < 2:
        print(f"[{set_name}] Only one class present in this set -- skipping metric computation.")
        return 0, 0, avg_loss, 0, 0

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    thresholds = np.arange(0.0, 1.0, 0.001)
    best_hter, best_apcer, best_bpcer, best_threshold = 1.0, 1.0, 1.0, 0.5

    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(all_labels, (all_scores >= t).astype(int), labels=[0, 1]).ravel()
        apcer = fp / (tn + fp + 1e-8)
        bpcer = fn / (fn + tp + 1e-8)
        hter = (apcer + bpcer) / 2

        if hter < best_hter:
            best_hter, best_apcer, best_bpcer, best_threshold = hter, apcer, bpcer, t

    auc = roc_auc_score(all_labels, all_scores)
    print(
        f"[{set_name}] Loss:{avg_loss:.4f} | HTER:{best_hter:.4f} | AUC:{auc:.4f} | "
        f"APCER:{best_apcer:.4f} | BPCER:{best_bpcer:.4f} | Threshold:{best_threshold:.3f}"
    )

    return best_hter, auc, avg_loss, best_apcer, best_bpcer

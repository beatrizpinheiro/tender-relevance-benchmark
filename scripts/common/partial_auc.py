"""Partial AUC restricted by recall (TPR), not by FPR.

`sklearn.metrics.roc_auc_score(max_fpr=...)` restricts the ROC curve to the
low-FPR region; here we need the opposite — the high-recall (TPR) region,
which is the part of the curve that matters for the review queue (whoever
clears the queue needs to catch almost all approved tenders, not just avoid
false positives). That's why the calculation is implemented here instead of
using scikit-learn directly.
"""

import numpy as np
from sklearn.metrics import roc_curve


def partial_auc_min_recall(y_true, y_score, min_recall: float = 0.8) -> float:
    """AUC of the ROC curve region with TPR >= min_recall, normalized to [0, 1].

    The region is the TPR(FPR) curve starting at the first point where the
    curve reaches `min_recall` (`roc_curve`'s `tpr` is non-decreasing, so
    `searchsorted` locates that point without needing interpolation — `tpr`
    has plateaus and isn't strictly increasing, which would break
    `np.interp`). Normalized by the maximum possible area from that point
    on — the height-1 rectangle — so a perfect classifier scores 1.0.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("y_true must contain both classes to compute AUC")

    fpr, tpr, _ = roc_curve(y_true, y_score)

    idx_cut = int(np.searchsorted(tpr, min_recall, side="left"))
    idx_cut = min(idx_cut, len(tpr) - 1)
    fpr_at_cut = fpr[idx_cut]

    area = float(np.trapz(tpr[idx_cut:], fpr[idx_cut:]))
    max_area = 1.0 - fpr_at_cut
    return area / max_area if max_area > 0 else 1.0

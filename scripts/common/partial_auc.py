"""Partial AUC restricted by recall (TPR), not by FPR.

`sklearn.metrics.roc_auc_score(max_fpr=...)` restricts the ROC curve to the
low-FPR region, averaging TPR over that FPR band; here we need the mirror
image — the high-recall (TPR) region, averaging specificity (1 - FPR) over
that TPR band — because what matters for the review queue is how many
false positives pile up once recall is already high (catching almost all
approved tenders), not how far up FPR climbs before TPR does anything.
That's why the calculation is implemented here instead of using
scikit-learn directly.

Fixed 16/11/2026: the previous version integrated TPR over the FPR axis
(`trapz(tpr[idx_cut:], fpr[idx_cut:]) / (1 - fpr_at_cut)`) instead of
specificity over the TPR axis. Because `roc_curve`'s TPR is non-decreasing,
every point from the TPR>=min_recall cutoff onward already has TPR close
to or above min_recall by construction — so that integral averaged out
close to 1.0 for *any* classifier, including random ones (empirically
~0.86-0.91 for pure noise), and the `max_area == 0` fallback returned a
literal 1.0 for a fully-tied score (e.g. a saturated LLM sigmoid), the
single worst case it should have scored near 0. Verified against the
corrected formula below: all-tied scores -> 0.0, random scores -> ~0.08,
a perfect separator -> 1.0.
"""

import numpy as np
from sklearn.metrics import roc_curve


def partial_auc_min_recall(y_true, y_score, min_recall: float = 0.8) -> float:
    """Average specificity (1 - FPR) over the ROC curve's TPR >= min_recall
    band, normalized to [0, 1] (1.0 = perfect separator in that band).

    Finds the band by index (`searchsorted` on `tpr`, which is
    non-decreasing) rather than interpolating at exactly `min_recall`:
    `tpr` has plateaus from tied scores, which breaks `np.interp`'s
    monotonicity assumption. The same reasoning as the original version,
    just applied to the right axis.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("y_true must contain both classes to compute AUC")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr

    idx_cut = int(np.searchsorted(tpr, min_recall, side="left"))
    idx_cut = min(idx_cut, len(tpr) - 1)

    tpr_band = tpr[idx_cut:]
    spec_band = specificity[idx_cut:]

    width = tpr_band[-1] - tpr_band[0]
    if width <= 0:
        # The band collapses to a single TPR value (e.g. min_recall is only
        # reached at the curve's very last point) — no interval to average
        # over, so the band's own specificity *is* the answer, not a
        # hardcoded "perfect" score.
        return float(spec_band[0])

    area = float(np.trapz(spec_band, tpr_band))
    return area / width

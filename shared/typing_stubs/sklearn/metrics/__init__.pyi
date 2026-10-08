"""Consumed metrics signatures verified against scikit-learn 1.5.0 source."""

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

@overload
def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: Literal["micro", "macro", "samples", "weighted"] = "macro",
    sample_weight: ArrayLike | None = None,
    max_fpr: float | None = None,
    multi_class: Literal["raise", "ovr", "ovo"] = "raise",
    labels: ArrayLike | None = None,
) -> float: ...
@overload
def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: None,
    sample_weight: ArrayLike | None = None,
    max_fpr: float | None = None,
    multi_class: Literal["raise", "ovr", "ovo"] = "raise",
    labels: ArrayLike | None = None,
) -> float | NDArray[np.float64]: ...
def cohen_kappa_score(
    y1: ArrayLike,
    y2: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    weights: Literal["linear", "quadratic"] | None = None,
    sample_weight: ArrayLike | None = None,
) -> float: ...

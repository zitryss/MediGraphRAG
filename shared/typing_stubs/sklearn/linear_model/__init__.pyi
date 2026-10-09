"""Consumed estimator methods verified against scikit-learn 1.5.0 source."""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LogisticRegression:
    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet"] | None = "l2",
        *,
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: dict[str | int | float, float]
        | Literal["balanced"]
        | None = None,
        random_state: int | np.random.RandomState | None = None,
        solver: Literal[
            "lbfgs",
            "liblinear",
            "newton-cg",
            "newton-cholesky",
            "sag",
            "saga",
        ] = "lbfgs",
        max_iter: int = 100,
        multi_class: Literal["auto", "ovr", "multinomial", "deprecated"] = "deprecated",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int | None = None,
        l1_ratio: float | None = None,
    ) -> None: ...
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32 | np.float64]: ...

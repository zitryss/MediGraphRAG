# Narrow installed-library typing interfaces

The `sklearn` declarations describe the interfaces consumed by maintained code,
verified against installed **scikit-learn 1.5.0** source. They are not a complete
replacement for upstream scikit-learn typing and contain no permissive fallback.
Both checkers resolve this directory explicitly; its `.pyi` files remain inside
the maintained `shared` audit root.

Sources inspected:

- `sklearn.metrics._ranking.roc_auc_score`: default averaging returns a scalar;
  `average=None` can return a per-class array (or a scalar for binary labels).
- `sklearn.metrics._classification.cohen_kappa_score`.
- `sklearn.linear_model._logistic.LogisticRegression`: constructor, `fit`, and
  `predict_proba`. The consumed feature inputs are dense array-like values. The
  probability dtype admits both float32 and float64, rather than assuming the
  default solver's float64 result for every solver.

Unsupported library surfaces remain errors. Extend declarations only from
verified installed source, not by adding `Any`, catch-all attributes, or
speculative overloads. Runtime dependencies and import behavior are unchanged.
Independent review still owns broader negative-contract and compatibility checks.

import hashlib
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, SupportsFloat, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray


class ScoredItem(Protocol):
    def get(
        self,
        key: Literal["score"],
        default: float,
        /,
    ) -> SupportsFloat | str | bytes | bytearray: ...


Item = TypeVar("Item")
Scored = TypeVar("Scored", bound=ScoredItem)


def _default_score(item: ScoredItem) -> float:
    return float(item.get("score", 0.0))


def compute_candidate_limit(
    max_chunks: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    *,
    hard_cap: int = 200,
) -> int:
    """Return the candidate-pool size for stochastic final-stage selection."""
    base: int = max(1, int(max_chunks))
    if float(retrieval_temperature or 0.0) <= 0.0:
        return base
    factor: int = max(1, int(shortlist_factor or 1))
    return max(base, min(base * factor, int(hard_cap)))


def stable_sample_seed(*parts: object) -> int:
    raw: str = "||".join("" if part is None else str(part) for part in parts)
    digest: bytes = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@overload
def select_ranked_subset(
    items: Sequence[Item],
    *,
    max_items: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    sample_id: int = 0,
    seed_parts: Sequence[object] | None = None,
    score_getter: Callable[[Item], float],
) -> list[Item]: ...


@overload
def select_ranked_subset(
    items: Sequence[Item],
    *,
    max_items: int,
    retrieval_temperature: Literal[0],
    shortlist_factor: int,
    sample_id: int = 0,
    seed_parts: Sequence[object] | None = None,
    score_getter: None = None,
) -> list[Item]: ...


@overload
def select_ranked_subset(
    items: Sequence[Scored],
    *,
    max_items: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    sample_id: int = 0,
    seed_parts: Sequence[object] | None = None,
    score_getter: None = None,
) -> list[Scored]: ...


def select_ranked_subset(
    items: Sequence[Item],
    *,
    max_items: int,
    retrieval_temperature: float,
    shortlist_factor: int,
    sample_id: int = 0,
    seed_parts: Sequence[object] | None = None,
    score_getter: Callable[[Item], float] | None = None,
) -> list[Item]:
    """Sample a top-k subset from a ranked list while preserving original order.

    When `retrieval_temperature <= 0`, returns the first `max_items` unchanged.
    Otherwise, samples from the top shortlist using a Gumbel-top-k perturbation
    over the item scores. The selected items are returned in their original
    ranked order so we isolate set changes from ordering changes.
    """
    ranked: list[Item] = list(items)
    k: int = max(1, int(max_items))
    if len(ranked) <= k:
        return ranked[:k]

    temp: float = float(retrieval_temperature or 0.0)
    if temp <= 0.0:
        return ranked[:k]

    candidate_limit: int = min(
        len(ranked),
        compute_candidate_limit(k, temp, shortlist_factor),
    )
    candidates: list[Item] = ranked[:candidate_limit]
    if len(candidates) <= k:
        return candidates[:k]

    # The stochastic no-callback overload requires ScoredItem. Preserve truthy
    # callback selection, including errors from malformed/default-scored inputs.
    get_score: Callable[[Item], float] = score_getter or cast(
        Callable[[Item], float],
        _default_score,
    )
    scores: NDArray[np.float64] = np.asarray(
        [float(get_score(item)) for item in candidates],
        dtype=float,
    )
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    temp = max(temp, 1e-6)
    logits: NDArray[np.float64] = scores / temp
    rng: np.random.Generator = np.random.default_rng(
        stable_sample_seed(*(seed_parts or ()), sample_id),
    )
    gumbels: NDArray[np.float64] = rng.gumbel(size=len(candidates))
    sampled_order: NDArray[np.intp] = np.argsort(-(logits + gumbels))
    selected: NDArray[np.intp] = np.sort(sampled_order[:k])
    return [candidates[idx] for idx in selected]

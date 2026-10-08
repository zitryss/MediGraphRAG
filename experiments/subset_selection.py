"""
Helpers for deterministic dataset question subset selection.

This module is dependency-light so it can be unit-tested without importing the
full experiment stack.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing_extensions import TypedDict


class SubsetIdentity(TypedDict):
    selection_key: str
    subset_hash: str
    subset_tag: str
    subset_id: str


class SubsetPayload(TypedDict, total=False):
    version: int
    dataset: str
    created_at: str
    selection_strategy: str
    subset_seed: int
    requested_num_samples: int | None
    available_question_count: int
    selection_count: int
    question_ids: list[str]
    selection_key: str
    subset_hash: str
    subset_tag: str
    subset_id: str


def sanitize_selection_token(value: str) -> str:
    """Create a filesystem-safe token."""
    token: str = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return token.strip("_") or "default"


def sample_size_token(num_samples: int | None) -> str:
    """Return a stable token for the requested sample count."""
    return "all" if num_samples is None else str(int(num_samples))


def selection_key(*, num_samples: int | None, subset_seed: int) -> str:
    """Return the stable key for a seeded subset specification."""
    return f"n{sample_size_token(num_samples)}_seed{int(subset_seed)}"


def selection_file_path(
    base_dir: Path,
    dataset_name: str,
    *,
    num_samples: int | None,
    subset_seed: int,
) -> Path:
    """Return the persisted selection path for one dataset/seed/sample-count subset."""
    safe_dataset: str = sanitize_selection_token(dataset_name)
    safe_key: str = sanitize_selection_token(
        selection_key(num_samples=num_samples, subset_seed=subset_seed)
    )
    return base_dir / f"{safe_dataset}__{safe_key}.json"


def _subset_hash(question_ids: Sequence[str]) -> str:
    """Short stable hash of the selected question IDs."""
    joined: bytes = "\n".join(str(question_id) for question_id in question_ids).encode()
    return hashlib.sha1(joined).hexdigest()[:12]


def subset_identity(
    *,
    dataset_name: str,
    question_ids: Sequence[str],
    num_samples: int | None,
    subset_seed: int,
) -> SubsetIdentity:
    """Return stable identifiers for a selected dataset subset."""
    key: str = selection_key(num_samples=num_samples, subset_seed=subset_seed)
    subset_hash: str = _subset_hash(question_ids)
    tag: str = f"{key}_h{subset_hash}"
    return {
        "selection_key": key,
        "subset_hash": subset_hash,
        "subset_tag": tag,
        "subset_id": f"{sanitize_selection_token(dataset_name)}__{tag}",
    }


def deterministic_subset_ids(
    ordered_record_ids: Sequence[str],
    num_samples: int | None,
    subset_seed: int,
) -> list[str]:
    """
    Choose a deterministic random subset while preserving original dataset order.

    We sample the IDs with a seeded RNG, then project that sampled set back onto
    the original ordered list. This keeps the chosen subset random while making
    evaluation order stable and easy to compare across runs.
    """
    ordered_ids: list[str] = [
        str(record_id) for record_id in ordered_record_ids if str(record_id)
    ]
    if not ordered_ids:
        return []

    if num_samples is None or num_samples >= len(ordered_ids):
        return list(ordered_ids)

    if num_samples <= 0:
        return []

    rng: random.Random = random.Random(int(subset_seed))
    sampled_ids: set[str] = set(rng.sample(ordered_ids, int(num_samples)))
    return [record_id for record_id in ordered_ids if record_id in sampled_ids]


@dataclass
class QuestionSubsetResolution:
    """Result of resolving a dataset question subset."""

    question_ids: list[str]
    payload: SubsetPayload
    created: bool
    warnings: list[str]


def _enrich_payload_identity(
    *,
    dataset_name: str,
    question_ids: Sequence[str],
    requested_num_samples: int | None,
    subset_seed: int,
    payload: SubsetPayload,
) -> SubsetPayload:
    """Backfill stable subset identity fields into the persisted payload."""
    identity: SubsetIdentity = subset_identity(
        dataset_name=dataset_name,
        question_ids=question_ids,
        num_samples=requested_num_samples,
        subset_seed=subset_seed,
    )
    payload = payload.copy()
    payload.setdefault("selection_key", identity["selection_key"])
    payload.setdefault("subset_hash", identity["subset_hash"])
    payload.setdefault("subset_tag", identity["subset_tag"])
    payload.setdefault("subset_id", identity["subset_id"])
    return payload


def resolve_question_subset(
    *,
    dataset_name: str,
    available_question_ids: Sequence[str],
    num_samples: int | None,
    subset_seed: int,
    selection_path: Path,
    require_existing: bool = False,
    force_resample: bool = False,
) -> QuestionSubsetResolution:
    """Load or create a persisted deterministic question subset."""
    ordered_ids: list[str] = [
        str(question_id) for question_id in available_question_ids if str(question_id)
    ]
    available_id_set: set[str] = set(ordered_ids)
    requested_num_samples: int | None = (
        None if num_samples is None else int(num_samples)
    )

    if selection_path.exists() and not force_resample:
        payload: SubsetPayload = json.loads(selection_path.read_text())
        saved_ids: list[str] = [
            str(question_id)
            for question_id in payload.get("question_ids", [])
            if str(question_id)
        ]
        if not saved_ids and ordered_ids:
            raise ValueError(
                f"Persisted selection at {selection_path} is missing question_ids."
            )

        missing_ids: list[str] = [
            question_id
            for question_id in saved_ids
            if question_id not in available_id_set
        ]
        if missing_ids:
            preview: str = ", ".join(missing_ids[:5])
            raise ValueError(
                f"Persisted selection at {selection_path} contains IDs not present in "
                f"the current dataset '{dataset_name}': {preview}"
            )

        warnings: list[str] = []
        saved_seed: int | None = payload.get("subset_seed")
        saved_requested: int | None = payload.get("requested_num_samples")
        if saved_seed is not None and int(saved_seed) != int(subset_seed):
            warnings.append(
                f"Persisted question subset for {dataset_name} uses subset_seed={saved_seed}; "
                f"requested subset_seed={subset_seed} will be ignored until you rebuild the KG."
            )
        if saved_requested != requested_num_samples:
            warnings.append(
                f"Persisted question subset for {dataset_name} uses num_samples={saved_requested}; "
                f"requested num_samples={requested_num_samples} will be ignored until you rebuild the KG."
            )

        payload = _enrich_payload_identity(
            dataset_name=dataset_name,
            question_ids=saved_ids,
            requested_num_samples=saved_requested,
            subset_seed=int(saved_seed if saved_seed is not None else subset_seed),
            payload=payload,
        )
        selection_path.write_text(json.dumps(payload, indent=2))

        return QuestionSubsetResolution(
            question_ids=saved_ids,
            payload=payload,
            created=False,
            warnings=warnings,
        )

    if require_existing and not force_resample:
        raise FileNotFoundError(
            f"No persisted question subset found for dataset '{dataset_name}' at {selection_path}. "
            f"Rebuild the KG to create one."
        )

    selected_ids: list[str] = deterministic_subset_ids(
        ordered_record_ids=ordered_ids,
        num_samples=num_samples,
        subset_seed=subset_seed,
    )
    payload = {
        "version": 1,
        "dataset": dataset_name,
        "created_at": datetime.now().isoformat(),
        "selection_strategy": "all"
        if len(selected_ids) == len(ordered_ids)
        else "seeded_sample",
        "subset_seed": int(subset_seed),
        "requested_num_samples": requested_num_samples,
        "available_question_count": len(ordered_ids),
        "selection_count": len(selected_ids),
        "question_ids": selected_ids,
    }
    payload = _enrich_payload_identity(
        dataset_name=dataset_name,
        question_ids=selected_ids,
        requested_num_samples=requested_num_samples,
        subset_seed=subset_seed,
        payload=payload,
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(payload, indent=2))
    return QuestionSubsetResolution(
        question_ids=selected_ids,
        payload=payload,
        created=True,
        warnings=[],
    )

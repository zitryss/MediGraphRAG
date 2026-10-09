#!/usr/bin/env python3
"""Validate the arXiv v1 cached-artifact package.

This is a packaging check only. It verifies cached files and, when the live
``paper/`` sources are present locally, manuscript wording, figure/table
references, and a local TeX build. It must not run experiments, call APIs,
rebuild KGs, or regenerate scientific results.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypeAlias

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from _hashlib import HASH

REPO: Path = Path(__file__).resolve().parents[1]
PACKAGE: Path = REPO / "reproducibility" / "arxiv_v1"
MANIFEST: Path = PACKAGE / "MANIFEST.json"
PAPER: Path = REPO / "paper"

BANNED_SOURCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "future-tense availability": re.compile(r"\bwill be released\b", re.IGNORECASE),
    "unqualified clinical certificate": re.compile(
        r"\bclinical certificate\b",
        re.IGNORECASE,
    ),
    "causal stress-test phrasing": re.compile(r"\bclearest mechanism\b", re.IGNORECASE),
    "deployment-strength certificate": re.compile(
        r"\bdeployable certificate\b",
        re.IGNORECASE,
    ),
}


JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


class ManifestFile(TypedDict, total=False):
    path: str
    role: str
    sha256: str
    size: int
    source_path: str


class Manifest(TypedDict, total=False):
    files: list[ManifestFile] | None
    created_at_utc: str
    description: str


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h: HASH = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> JSONValue:
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST.relative_to(REPO)}")
    try:
        manifest: JSONValue = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return manifest
    except json.JSONDecodeError as exc:
        fail(f"manifest is not valid JSON: {exc}")


def check_manifest_files(doc: Manifest | JSONValue) -> None:
    files: list[ManifestFile] | JSONValue = doc.get("files")
    if not isinstance(files, list) or not files:
        fail("manifest must contain a non-empty 'files' list")

    required_roles: set[str] = {
        "selection_ids",
        "headline_results",
        "analysis_output",
        "gps_replay",
    }
    roles: set[JSONValue] = {item.get("role") for item in files}
    missing_roles: list[str] = sorted(required_roles - roles)
    if missing_roles:
        fail(f"manifest missing required roles: {missing_roles}")

    for item in files:
        rel: JSONValue = item.get("path")
        expected_hash: JSONValue = item.get("sha256")
        expected_size: JSONValue = item.get("size")
        if not rel or not expected_hash or expected_size is None:
            fail(f"manifest entry missing path/sha256/size: {item}")
        path: Path = REPO / rel
        if not path.exists():
            fail(f"manifest file missing: {rel}")
        if path.stat().st_size != expected_size:
            fail(f"size mismatch for {rel}: {path.stat().st_size} != {expected_size}")
        actual_hash: str = sha256(path)
        if actual_hash != expected_hash:
            fail(f"sha256 mismatch for {rel}: {actual_hash} != {expected_hash}")


def paper_sources() -> list[Path]:
    return sorted(PAPER.glob("*.tex")) + [PAPER / "references.bib"]


def check_manuscript_wording() -> None:
    for path in paper_sources():
        if not path.exists():
            continue
        text: str = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in BANNED_SOURCE_PATTERNS.items():
            match: re.Match[str] | None = pattern.search(text)
            if match:
                line: int = text.count("\n", 0, match.start()) + 1
                fail(f"{label} found in {path.relative_to(REPO)}:{line}")


def strip_latex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        escaped: bool = False
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def check_figure_inputs() -> None:
    text: str = "\n".join(
        strip_latex_comments(path.read_text(encoding="utf-8", errors="replace"))
        for path in PAPER.glob("*.tex")
    )

    input_refs: list[str] = re.findall(r"\\input\{figures/([^}]+)\}", text)
    for ref in sorted(set(input_refs)):
        candidate: Path = PAPER / "figures" / ref
        if candidate.suffix:
            if not candidate.exists():
                fail(f"missing figure/table input: {candidate.relative_to(REPO)}")
        elif not candidate.with_suffix(".tex").exists():
            fail(
                f"missing figure/table input: {candidate.with_suffix('.tex').relative_to(REPO)}",
            )

    graphic_refs: list[str] = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}",
        text,
    )
    for ref in sorted(set(graphic_refs)):
        candidate = PAPER / "figures" / ref
        if candidate.suffix:
            if not candidate.exists():
                fail(f"missing graphic: {candidate.relative_to(REPO)}")
        else:
            found: bool = any(
                candidate.with_suffix(ext).exists()
                for ext in (".pdf", ".png", ".jpg", ".jpeg")
            )
            if not found:
                fail(f"missing graphic for figures/{ref} with pdf/png/jpg extension")


def run_tectonic() -> None:
    cmd: list[str] = ["tectonic", "main_arxiv.tex"]
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        cwd=PAPER,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        fail("tectonic build failed")


def main() -> None:
    doc: JSONValue = load_manifest()
    check_manifest_files(doc)

    if not PAPER.is_dir():
        print(
            "arXiv package check passed (paper sources not present locally; skipped paper checks)",
        )
        return

    check_manuscript_wording()
    check_figure_inputs()
    run_tectonic()
    print("arXiv package check passed")


if __name__ == "__main__":
    main()

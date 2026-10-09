"""
Prepare a shared BioASQ retrieval corpus from PubMed.

This converts BioASQ's question-local gold file (which only contains question
metadata, PubMed URLs, and expert snippets) into a shared abstract corpus that
can be indexed fairly by both vanilla RAG and KG-RAG.

Output format: JSONL with one record per PMID, e.g.
    {"pmid": "34022131", "title": "...", "abstract": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from typing_extensions import TypedDict


class PubMedAbstract(TypedDict):
    pmid: str
    title: str
    abstract: str


logger: logging.Logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_BIOASQ_PATH: Path = Path(
    "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
)
DEFAULT_OUTPUT_PATH: Path = Path("MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl")
PMID_RE: re.Pattern[str] = re.compile(r"/pubmed/(\d+)")


def extract_pmid_from_url(url: str) -> str | None:
    """Extract a PMID from a BioASQ PubMed URL."""
    text: str = str(url or "").strip()
    match: re.Match[str] | None = PMID_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return None


def load_bioasq_pmids(bioasq_path: Path) -> list[str]:
    """Collect unique PMIDs referenced by a BioASQ golden file."""
    data: object = json.loads(bioasq_path.read_text())
    questions: object = data.get("questions", []) if isinstance(data, dict) else []

    pmids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            continue
        for url in question.get("documents", []) or []:
            pmid: str | None = extract_pmid_from_url(str(url))
            if pmid:
                pmids.add(pmid)
        for snippet in question.get("snippets", []) or []:
            if not isinstance(snippet, dict):
                continue
            pmid = extract_pmid_from_url(str(snippet.get("document", "")))
            if pmid:
                pmids.add(pmid)

    return sorted(pmids, key=int)


def _stringify_xml_text(elem: ET.Element | None) -> str:
    """Flatten an XML element into readable text."""
    if elem is None:
        return ""
    text: str = "".join(elem.itertext())
    return re.sub(r"\s+", " ", text).strip()


def parse_pubmed_article_xml(xml_text: str) -> dict[str, PubMedAbstract]:
    """Parse an EFetch PubMedArticleSet XML response into PMID -> record."""
    root: ET.Element = ET.fromstring(xml_text)
    records: dict[str, PubMedAbstract] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid: str = _stringify_xml_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue

        article_title: str = _stringify_xml_text(
            article.find(".//Article/ArticleTitle"),
        )
        abstract_nodes: list[ET.Element] = article.findall(
            ".//Article/Abstract/AbstractText",
        )
        abstract_parts: list[str] = []
        for node in abstract_nodes:
            label: str = str(node.attrib.get("Label", "")).strip()
            part: str = _stringify_xml_text(node)
            if not part:
                continue
            if label:
                abstract_parts.append(f"{label}: {part}")
            else:
                abstract_parts.append(part)
        abstract: str = "\n".join(abstract_parts).strip()

        records[pmid] = {
            "pmid": pmid,
            "title": article_title,
            "abstract": abstract,
        }

    return records


def fetch_pubmed_batch(
    pmids: Sequence[str],
    *,
    email: str | None = None,
    api_key: str | None = None,
    tool: str = "ontographrag",
    timeout_seconds: float = 60.0,
) -> dict[str, PubMedAbstract]:
    """Fetch one PMID batch from PubMed via EFetch XML."""
    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    response: requests.Response = requests.get(
        f"{NCBI_EUTILS_BASE}/efetch.fcgi",
        params=params,
        timeout=timeout_seconds,
        headers={"User-Agent": f"{tool}/1.0"},
    )
    response.raise_for_status()
    return parse_pubmed_article_xml(response.text)


def load_existing_corpus(output_path: Path) -> dict[str, object]:
    """Load an existing JSONL corpus file so reruns can resume incrementally."""
    existing: dict[str, object] = {}
    if not output_path.exists():
        return existing

    with output_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid: str = str(record.get("pmid", "")).strip()
            if pmid:
                existing[pmid] = record
    return existing


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])


def build_bioasq_shared_corpus(
    *,
    bioasq_path: Path = DEFAULT_BIOASQ_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    email: str | None = None,
    api_key: str | None = None,
    batch_size: int = 100,
    sleep_seconds: float = 0.34,
    max_pmids: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Build a shared BioASQ abstract corpus.

    Returns:
        (num_records_written, num_missing_records)
    """
    if not bioasq_path.exists():
        raise FileNotFoundError(f"BioASQ file not found: {bioasq_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pmids: list[str] = load_bioasq_pmids(bioasq_path)
    if max_pmids is not None:
        pmids = pmids[:max_pmids]

    existing: dict[str, object] = {} if overwrite else load_existing_corpus(output_path)
    pending: list[str] = [pmid for pmid in pmids if pmid not in existing]

    logger.info(
        "BioASQ corpus prep: %d PMIDs total, %d already cached, %d pending",
        len(pmids),
        len(existing),
        len(pending),
    )

    fetched: dict[str, object] = dict(existing)
    missing: set[str] = set()

    for batch_num, batch in enumerate(_chunked(pending, batch_size), start=1):
        logger.info("Fetching PubMed batch %d (%d PMIDs)", batch_num, len(batch))
        try:
            records: dict[str, PubMedAbstract] = fetch_pubmed_batch(
                batch,
                email=email,
                api_key=api_key,
            )
        except Exception as e:
            logger.error("Failed to fetch batch %d: %s", batch_num, e)
            raise

        fetched.update(records)
        batch_missing: set[str] = set(batch) - set(records.keys())
        if batch_missing:
            missing.update(batch_missing)
            logger.warning(
                "Batch %d returned no abstract/title for %d PMIDs",
                batch_num,
                len(batch_missing),
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    ordered_records: list[object] = []
    for pmid in pmids:
        record: object = fetched.get(pmid)
        if not record:
            missing.add(pmid)
            continue
        ordered_records.append(record)

    with output_path.open("w") as f:
        for record in ordered_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    logger.info(
        "Wrote %d BioASQ abstract records to %s (%d missing)",
        len(ordered_records),
        output_path,
        len(missing),
    )
    return len(ordered_records), len(missing)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build a shared PubMed abstract corpus for BioASQ retrieval benchmarking.",
    )
    parser.add_argument(
        "--bioasq-path",
        type=Path,
        default=DEFAULT_BIOASQ_PATH,
        help="Path to the BioASQ golden JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for fetched PubMed abstracts.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.getenv("NCBI_EMAIL") or os.getenv("ENTREZ_EMAIL"),
        help="Contact email sent to NCBI E-utilities.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("NCBI_API_KEY"),
        help="Optional NCBI API key for higher rate limits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="PMIDs to fetch per EFetch request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.34,
        help="Delay between EFetch batches to stay polite to NCBI.",
    )
    parser.add_argument(
        "--max-pmids",
        type=int,
        default=None,
        help="Optional cap for smoke-testing corpus preparation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Ignore any existing output JSONL and rebuild from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable info-level logging.",
    )
    args: argparse.Namespace = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    build_bioasq_shared_corpus(
        bioasq_path=args.bioasq_path,
        output_path=args.output,
        email=args.email,
        api_key=args.api_key,
        batch_size=max(1, args.batch_size),
        sleep_seconds=max(0.0, args.sleep_seconds),
        max_pmids=args.max_pmids,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

import dataclasses
import json
from pathlib import Path

import structlog

from docuseek.ingestion.cleaners import CleanDocument, clean
from docuseek.ingestion.scrapers.base import RawDocument
from docuseek.ingestion.scrapers.huggingface import HuggingFaceScraper

logger = structlog.get_logger()


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


SCRAPERS = {
    "huggingface": HuggingFaceScraper,
}


def run_ingestion(
    scraper_names: list[str] | None = None, force_rescrape: bool = False
) -> list[CleanDocument]:
    """
    Full ingestion pipeline: scrape -> clean -> save to disk.

    Scrapes the requested sources, cleans each document, discards
    documents that are too short, and saves the results as JSONL
    to data/processed/{source}.jsonl — one file per source.

    Args:
        scraper_names: which sources to scrape, e.g. ["huggingface"].
                       If None, runs all available scrapers.
        force_rescrape: whether to re-scrape sources even if raw data exists.

    Returns:
        flat list of all CleanDocuments produced
    """
    if scraper_names is None:
        scraper_names = list(SCRAPERS.keys())

    all_clean_docs = []
    for name in scraper_names:
        if name not in SCRAPERS:
            logger.warning("unknown_scraper", name=name)
            continue

        raw_path = RAW_DATA_DIR / f"{name}.jsonl"
        if raw_path.exists() and not force_rescrape:
            logger.info("loading_raw_from_disk", source=name, path=str(raw_path))
            raw_docs = load_raw_jsonl(raw_path)
            logger.info("loaded_raw_from_disk", source=name, documents=len(raw_docs))
        else:
            logger.info("scraping_start", source=name)
            raw_docs = SCRAPERS[name]().scrape()
            _save_raw_jsonl(raw_docs, raw_path)
            logger.info("scraping_complete", source=name, documents=len(raw_docs))

        clean_docs = list(filter(None, (clean(doc) for doc in raw_docs if doc is not None)))
        all_clean_docs.extend(clean_docs)
        _save_jsonl(clean_docs, PROCESSED_DATA_DIR / f"{name}.jsonl")
        logger.info("ingestion_complete", source=name, documents=len(clean_docs))
    return all_clean_docs


def _save_raw_jsonl(documents: list[RawDocument], path: Path) -> None:
    """
    Save raw scraped documents to disk before cleaning.
    Allows re-running the cleaning step without re-scraping.

    Args:
        documents: list of RawDocuments to save
        path: destination file path e.g. data/raw/huggingface.jsonl
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")


def load_raw_jsonl(path: Path) -> list[RawDocument]:
    """
    Load RawDocuments from a previously saved raw JSONL file.

    Args:
        path: path to the JSONL file
    Returns:
        list of RawDocuments
    """
    with path.open("r", encoding="utf-8") as f:
        return [RawDocument(**json.loads(line.strip())) for line in f]


def _save_jsonl(documents: list[CleanDocument], path: Path) -> None:
    """
    Save a list of CleanDocuments to a JSONL file.
    Creates parent directories if they do not exist.
    Each line is a JSON object with all CleanDocument fields.

    Args:
        documents: documents to save
        path: destination file path, e.g. data/processed/huggingface.jsonl
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")


def load_jsonl(path: Path) -> list[CleanDocument]:
    """
    Load CleanDocuments from a JSONL file previously saved by _save_jsonl.

    Args:
        path: path to the JSONL file
    Returns:
        list of CleanDocuments
    """
    documents = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(CleanDocument(**json.loads(line.strip())))
    return documents

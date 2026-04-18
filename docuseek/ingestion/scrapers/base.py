from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RawDocument:
    """A single scraped page before any cleaning."""

    url: str
    title: str
    content: str  # raw HTML or markdown
    source: str  # "huggingface" | "pytorch" | "vllm"
    metadata: dict  # anything extra: section, version, ...


class BaseScraper(ABC):
    @abstractmethod
    def scrape(self) -> list[RawDocument]:
        """
        Scrape the full documentation source.
        Returns a list of RawDocuments, one per page.
        """
        ...

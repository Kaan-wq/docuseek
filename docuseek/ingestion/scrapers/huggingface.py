import httpx
import structlog
from tqdm import tqdm

from docuseek.config import settings
from docuseek.ingestion.scrapers.base import BaseScraper, RawDocument

logger = structlog.get_logger()


HEADERS = {"Authorization": f"Bearer {settings.github_token}"}


LIBRARIES: dict[str, str] = {
    "transformers": "docs/source/en",
    "diffusers": "docs/source/en",
    "datasets": "docs/source",
    "peft": "docs/source",
    "tokenizers": "docs/source",
    "accelerate": "docs/source",
}


class HuggingFaceScraper(BaseScraper):
    def __init__(self, libraries: dict[str, str] = LIBRARIES) -> None:
        """
        Args:
            libraries: mapping of library name to its docs path on GitHub
        """
        self.libraries = libraries

    def scrape(self) -> list[RawDocument]:
        """
        Scrape all pages for all configured libraries.
        Calls _scrape_library for each and flattens results.
        """
        all_docs = []
        # limit size of bar to quarter of screen
        for library in tqdm(
            self.libraries, desc="Scraping libraries", unit="lib", maxinterval=0.5, ncols=80
        ):
            try:
                docs = self._scrape_library(library)
                all_docs.extend(docs)
                tqdm.write(f"  {library}: {len(docs)} documents")
            except Exception as e:
                logger.warning("scraping_failed", library=library, error=str(e))
        return all_docs

    def _scrape_library(self, library: str) -> list[RawDocument]:
        """
        Fetch the list of doc pages for one library from the
        GitHub contents API, then fetch each page's markdown.
        Base URL pattern:
          https://raw.githubusercontent.com/huggingface/{library}/main/docs/source/en/
        Args:
            library: e.g. "transformers"
        Returns:
            list of RawDocument, one per .md or .mdx file found
        """
        docs_path = self.libraries[library]
        base_url = f"https://api.github.com/repos/huggingface/{library}/contents/{docs_path}"
        return self._scrape_directory(base_url, library)

    def _scrape_directory(self, url: str, library: str) -> list[RawDocument]:
        """
        Recursively scrape all .md and .mdx files under a directory.
        If an entry is a directory (download_url is None), recurse into it.
        If an entry is a .md or .mdx file, fetch and return it.
        All other file types are skipped.
        Args:
            api_url: GitHub contents API URL for this directory
            library: passed through to RawDocument.source
        Returns:
            flat list of RawDocument from this directory and all subdirectories
        """
        entries = httpx.get(url=url, headers=HEADERS)
        if entries.is_error:
            logger.warning("fetch_failed", url=url, status_code=entries.status_code)
            return []
        entries = entries.json()

        documents = []
        for entry in entries:
            if entry["type"] == "dir":
                documents.extend(self._scrape_directory(entry["url"], library))
            elif entry["name"].endswith((".md", ".mdx")):
                doc = self._fetch_page(entry["download_url"], library)
                if doc:
                    documents.append(doc)
        return documents

    def _fetch_page(self, url: str, library: str) -> RawDocument | None:
        """
        Fetch a single markdown page and wrap it in a RawDocument.
        Returns None if the request fails or content is empty.
        Args:
            url: raw GitHub URL to the .md or .mdx file
            library: used to populate RawDocument.source
        """
        response = httpx.get(url=url, headers=HEADERS)
        if response.is_error:
            logger.warning("fetch_failed", url=url, status_code=response.status_code)
            return None
        if not response.text.strip():
            logger.warning("fetch_empty", url=url)
            return None
        content = response.text

        docs_path = self.libraries[library]

        return RawDocument(
            url=url,
            title=url.rsplit("/", maxsplit=1)[-1],  # crude title extraction from URL
            content=content,
            source=library,
            metadata={
                "path": url.rsplit(f"/{docs_path}/", maxsplit=1)[-1],
                "library": library,
            },
        )

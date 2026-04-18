import re
from dataclasses import dataclass

import structlog

from docuseek.ingestion.scrapers.base import RawDocument

logger = structlog.get_logger()

MIN_CONTENT_LENGTH = 200


@dataclass
class CleanDocument:
    url: str
    title: str
    content: str
    source: str
    metadata: dict


def clean(doc: RawDocument) -> CleanDocument | None:
    """
    Clean a raw MDX/MD document scraped from HuggingFace documentation.

    Steps applied in order:
      1. Strip HTML comments (copyright headers, etc.)
      2. Strip YAML frontmatter (--- delimited blocks)
      3. Strip HuggingFace-specific directives ([[open-in-colab]], etc.)
      4. Strip MDX import/export statements
      5. Strip JSX/HTML tags, preserving text content
      6. Normalize whitespace
      7. Discard if remaining content is shorter than MIN_CONTENT_LENGTH

    Code blocks are intentionally preserved — they are high-value content
    for technical documentation retrieval.

    Args:
        doc: raw document from the scraper
    Returns:
        CleanDocument if the document has enough content, None otherwise
    """
    text = doc.content

    text = _strip_html_comments(text)
    text = _strip_frontmatter(text)
    text = _strip_hf_directives(text)
    text = _strip_mdx_statements(text)
    text = _strip_jsx_tags(text)
    text = _normalize_whitespace(text)

    if len(text) < MIN_CONTENT_LENGTH:
        logger.debug(
            "document_discarded",
            url=doc.url,
            reason="too_short",
            length=len(text),
        )
        return None

    return CleanDocument(
        url=doc.url,
        title=_extract_title(text, doc.title),
        content=text,
        source=doc.source,
        metadata=doc.metadata,
    )


def _strip_html_comments(text: str) -> str:
    """
    Remove HTML comments including multi-line copyright headers.
    Example: <!-- Copyright 2024 The HuggingFace Team. All rights reserved. -->
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter delimited by --- at the start of the file.
    Only strips if the file starts with ---, handles optional whitespace.
    Example:
        ---
        title: My Page
        ---
    """
    return re.sub(r"^\s*---.*?---\s*", "", text, flags=re.DOTALL)


def _strip_hf_directives(text: str) -> str:
    """
    Remove HuggingFace-specific MDX directives that are structural noise.
    Examples:
        [[open-in-colab]]
        <Youtube id="..." />
    """
    text = re.sub(r"\[\[.*?\]\]", "", text)
    return text


def _strip_mdx_statements(text: str) -> str:
    """
    Remove MDX import and export lines.
    These are JavaScript module statements that have no semantic value
    for retrieval.
    Examples:
        import Tip from "@huggingface/doc-builder/...";
        export const metadata = { title: "..." };
    """
    text = re.sub(r"^import\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^export\s+.*$", "", text, flags=re.MULTILINE)
    return text


def _strip_jsx_tags(text: str) -> str:
    """
    Remove JSX and HTML tags while preserving their text content.
    Self-closing tags (no content) are discarded entirely.
    Block tags with content keep the content, drop the tags.

    Examples:
        <Tip>some useful text</Tip>     -> "some useful text"
        <img src="..." />               -> ""
        <a href="...">click here</a>    -> "click here"

    Code blocks are protected from this step — content inside ```...```
    is not touched.
    """
    # protect code blocks by temporarily replacing them
    code_blocks: list[str] = []

    def stash_code_block(match: re.Match) -> str:
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text = re.sub(r"```.*?```", stash_code_block, text, flags=re.DOTALL)

    # remove self-closing tags
    text = re.sub(r"<[A-Za-z][^>]*/\s*>", "", text)

    # remove open/close tag pairs, keep inner text
    text = re.sub(r"<[A-Za-z][^>]*>(.*?)</[A-Za-z]+>", r"\1", text, flags=re.DOTALL)

    # remove any remaining lone open or close tags
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)

    # restore code blocks
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)

    return text


def _normalize_whitespace(text: str) -> str:
    """
    Collapse 3 or more consecutive newlines into exactly 2.
    Strip leading and trailing whitespace.
    This preserves paragraph structure (double newline) without
    leaving large empty gaps.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(text: str, fallback: str) -> str:
    """
    Extract the first H1 heading from the cleaned content as the title.
    Falls back to the filename if no H1 is found.
    Example:
        # My Document Title  -> "My Document Title"
    """
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback.replace(".md", "").replace(".mdx", "")

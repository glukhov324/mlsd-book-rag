"""EPUB → главы (spine) → чанки RecursiveCharacter внутри каждой главы."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub
from langchain_text_splitters import RecursiveCharacterTextSplitter

RU_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "… ", "; ", " ", ""]


@dataclass(frozen=True)
class Chapter:
    book_title: str
    chapter_index: int
    chapter_href: str
    chapter_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    book_title: str
    chapter_index: int
    chapter_id: str
    chunk_index: int
    global_chunk_index: int
    text: str
    char_start: int
    char_end: int

    @property
    def chunk_id(self) -> str:
        return f"{self.chapter_id}:{self.chunk_index}"


def html_to_text(html: bytes) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_epub_chapters(
    epub_path: str | Path, *, min_chars: int = 50
) -> list[Chapter]:
    path = Path(epub_path)
    book = epub.read_epub(str(path))
    meta = book.get_metadata("DC", "title")
    book_title = meta[0][0] if meta else path.stem

    chapters: list[Chapter] = []
    chapter_index = 0
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        text = html_to_text(item.get_content())
        if len(text) < min_chars:
            continue
        href = item.get_name() or item_id
        chapter_id = Path(href).stem
        chapters.append(
            Chapter(
                book_title=book_title,
                chapter_index=chapter_index,
                chapter_href=href,
                chapter_id=chapter_id,
                text=text,
            )
        )
        chapter_index += 1
    return chapters


def chunk_chapters(
    chapters: list[Chapter],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or RU_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Chunk] = []
    global_chunk_index = 0
    for ch in chapters:
        parts = splitter.split_text(ch.text)
        pos = 0
        for chunk_index, piece in enumerate(parts):
            start = ch.text.find(piece, pos)
            if start < 0:
                start = pos
            end = start + len(piece)
            pos = max(pos, end - chunk_overlap)
            chunks.append(
                Chunk(
                    book_title=ch.book_title,
                    chapter_index=ch.chapter_index,
                    chapter_id=ch.chapter_id,
                    chunk_index=chunk_index,
                    global_chunk_index=global_chunk_index,
                    text=piece,
                    char_start=start,
                    char_end=end,
                )
            )
            global_chunk_index += 1
    return chunks


def split_epub(
    epub_path: str | Path,
    *,
    min_chars: int = 50,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    return chunk_chapters(
        load_epub_chapters(epub_path, min_chars=min_chars),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

from __future__ import annotations

from enum import Enum
from typing import Optional

from corpus.datasets.base import WikipediaDataset


class DefaultWikipediaUrls(Enum):
    TEXT = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-04-01/xml/bzip2"
    BOOK = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwikibooks/2026-04-01/xml/bzip2"
    VOYAGE = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwikivoyage/2026-04-01/xml/bzip2"
    SOURCE = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwikisource/2026-04-01/xml/bzip2"
    UNIVERSITY = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwikiversity/2026-04-01/xml/bzip2"


class WikipediaText(WikipediaDataset):
    def __init__(
        self,
        wiki_dump_url: Optional[str] = None,
        save_dir="dataset",
        dataset_name="wikipedia_text_eng",
    ):
        super().__init__(
            base_url=wiki_dump_url
            if wiki_dump_url
            else DefaultWikipediaUrls.TEXT.value,
            name=dataset_name,
            save_dir=save_dir,
        )


class WikipediaBooks(WikipediaDataset):
    def __init__(
        self,
        wiki_dump_url: Optional[str] = None,
        save_dir="dataset",
        dataset_name="wikipedia_books_eng",
    ):
        super().__init__(
            base_url=wiki_dump_url
            if wiki_dump_url
            else DefaultWikipediaUrls.BOOK.value,
            name=dataset_name,
            save_dir=save_dir,
        )


class WikipediaVoyage(WikipediaDataset):
    def __init__(
        self,
        wiki_dump_url: Optional[str] = None,
        save_dir="dataset",
        dataset_name="wikipedia_voyage_eng",
    ):
        super().__init__(
            base_url=wiki_dump_url
            if wiki_dump_url
            else DefaultWikipediaUrls.VOYAGE.value,
            name=dataset_name,
            save_dir=save_dir,
        )


class WikipediaSource(WikipediaDataset):
    def __init__(
        self,
        wiki_dump_url: Optional[str] = None,
        save_dir="dataset",
        dataset_name="wikipedia_source_eng",
    ):
        super().__init__(
            base_url=wiki_dump_url
            if wiki_dump_url
            else DefaultWikipediaUrls.SOURCE.value,
            name=dataset_name,
            save_dir=save_dir,
        )


class WikipediaUniversity(WikipediaDataset):
    def __init__(
        self,
        wiki_dump_url: Optional[str] = None,
        save_dir="dataset",
        dataset_name="wikipedia_university_eng",
    ):
        super().__init__(
            base_url=wiki_dump_url
            if wiki_dump_url
            else DefaultWikipediaUrls.UNIVERSITY.value,
            name=dataset_name,
            save_dir=save_dir,
        )

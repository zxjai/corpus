from corpus.datasets.base import WikipediaDataset
from corpus.datasets.wikipedia import (
    WikipediaBooks,
    WikipediaSource,
    WikipediaText,
    WikipediaUniversity,
    WikipediaVoyage,
)

BASE_URL = "https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-04-01/xml/bzip2"


def test_bulk():
    dataset = WikipediaDataset(base_url=BASE_URL)
    dataset.download_bulk()


def test_stream():
    dataset = WikipediaDataset(base_url=BASE_URL)
    dataset.remote_open()


def test_inspect():
    dataset = WikipediaDataset(base_url=BASE_URL)
    dataset.inspect()


def test_t():
    # texts
    dataset = WikipediaText()
    dataset.download_bulk()
    dataset.inspect()


def test_b():
    # books
    dataset = WikipediaBooks()
    dataset.download_bulk()
    dataset.inspect()


def test_v():
    # voyage
    dataset = WikipediaVoyage()
    dataset.download_bulk()
    dataset.inspect()


def test_s():
    # source
    dataset = WikipediaSource()
    dataset.download_bulk()
    dataset.inspect()


def test_u():
    # source
    dataset = WikipediaUniversity()
    dataset.download_bulk()
    dataset.inspect()

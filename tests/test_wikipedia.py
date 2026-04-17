from corpus.datasets.base import WikipediaDataset

BASE_URL = 'https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-04-01/xml/bzip2'

def test_bulk():
    dataset = WikipediaDataset(base_url=BASE_URL)
    dataset.download_bulk()

def test_stream():
    dataset = WikipediaDataset(base_url=BASE_URL)
    dataset.remote_open()
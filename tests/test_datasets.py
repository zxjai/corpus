from corpus.datasets.base import WikipediaDataset
from corpus.datasets.library import InstitutionalBooks, Poems, ProjectGutenberg


def test_pg():
    dataset = ProjectGutenberg()
    # dataset.download_bulk()
    dataset.inspect()


def test_poems():
    dataset = Poems()
    # dataset.download_bulk()
    dataset.inspect()


def test_ib():
    dataset = InstitutionalBooks()
    # dataset.download_bulk()
    dataset.inspect()

def test_wiki():
    base_url  = 'https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-04-01/xml/bzip2/'
    dataset = WikipediaDataset(base_url=base_url)
    dataset.download_single_shard()
    # url = 'https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-04-01/xml/bzip2/enwiki-2026-04-01-p23970571p28987923.xml.bz2'
    # dataset.remote_open(url)
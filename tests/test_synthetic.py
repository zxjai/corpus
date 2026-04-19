from corpus.datasets.synthetic import FinePhrase


def test_fp():
    dataset = FinePhrase()
    dataset.download_bulk()

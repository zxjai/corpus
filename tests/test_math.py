from corpus.datasets.math import MathNet


def test_mn():
    dataset = MathNet()
    dataset.download_bulk()

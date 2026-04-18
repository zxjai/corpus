from corpus.datasets.ensemble import DCLM, Dolma, NemotronCC


def test_dolma():
    dataset = Dolma()
    dataset.composition()
    dataset.download_subset("Books")


def test_dclm():
    dataset = DCLM()
    dataset.download_bulk(max_files=4)
    dataset.inspect()


def test_nemotron_cc():
    dataset = NemotronCC()
    # dataset.download_bulk(max_files=4)
    dataset.ls()

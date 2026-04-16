from corpus.datasets.library import InstitutionalBooks, ProjectGutenberg


def test_pg():
    dataset = ProjectGutenberg()
    # dataset.download_bulk()
    dataset.inspect()


def test_ib():
    dataset = InstitutionalBooks()
    # dataset.download_bulk()
    dataset.inspect()

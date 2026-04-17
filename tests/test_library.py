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
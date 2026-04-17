from corpus.datasets.code import GitHubDiscovery, StackV2


def test_gd():
    gd = GitHubDiscovery()
    gd.download_day()
    gd.star_frequency()


def test_stack():
    dataset = StackV2()
    dataset.ls_all()

from corpus.datasets.code import GitHubDiscovery


def test_gd():
    gd = GitHubDiscovery()
    gd.download_day()
    gd.star_frequency()

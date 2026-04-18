from corpus.datasets.code import GitHubDiscovery, StackV2


def test_gd():
    gd = GitHubDiscovery()
    gd.download_day()
    gd.star_frequency()
    # gd.most_starred_by_lang(lang="Python")


def test_stack():
    dataset = StackV2()
    # dataset.ls_all()
    dataset.ls(show_num=3)

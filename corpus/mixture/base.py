class EmbeddingModel:
    def __init__(self, repo_id: str, name: str, save_dir: str):
        self.repo_id = repo_id
        self.save_dir = save_dir
        self.name = name


class GuardrailModel:
    def __init__(self, repo_id: str, name: str, save_dir: str):
        self.repo_id = repo_id
        self.save_dir = save_dir
        self.name = name

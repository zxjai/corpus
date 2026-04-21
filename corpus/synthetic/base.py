from huggingface_hub import snapshot_download
from pathlib import Path


class SyntheticDataModel:
    def __init__(self, repo_id: str, name: str, save_dir: str):
        self.repo_id = repo_id
        self.save_dir = Path(save_dir) / name
        self.name = name

    def download(self, max_workers=8):
        snapshot_download(
            cache_dir=self._cache_dir(),
            local_dir=self._download_dir(),
            repo_type="model",
            repo_id=self.repo_id,
            max_workers=max_workers,
        )

    def _cache_dir(self) -> Path:
        p = self.save_dir / ".cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _download_dir(self) -> Path:
        p = self.save_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


class GPTOSS(SyntheticDataModel):
    def __init__(self, save_dir: str = "model", use_small: bool = True):
        self.num_param = 20 if use_small else 120
        super().__init__(
            repo_id=f"openai/gpt-oss-{self.num_param}b",
            name=f"gptoss_{self.num_param}b",
            save_dir=save_dir,
        )

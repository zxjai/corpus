from corpus.synthetic.base import SyntheticDataModel


class GPTOSS(SyntheticDataModel):
    def __init__(self, save_dir: str = "model", use_small: bool = True):
        self.num_param = 20 if use_small else 120
        super().__init__(
            repo_id=f"openai/gpt-oss-{self.num_param}b",
            name=f"gptoss_{self.num_param}b",
            save_dir=save_dir,
        )


class Qwen3(SyntheticDataModel):
    def __init__(self, save_dir: str = "model"):
        super().__init__(
            repo_id="Qwen/Qwen3-1.7B",
            name="qwen3_1.7b",
            save_dir=save_dir,
        )

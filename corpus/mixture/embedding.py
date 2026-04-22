from corpus.mixture.base import EmbeddingModel
from sentence_transformers import SentenceTransformer
from loguru import logger


class QwenVLEmbed(EmbeddingModel):
    def __init__(self, save_dir="models", use_small=True):
        self.model_size = 2 if use_small else 8
        super().__init__(
            repo_id=f"Qwen/Qwen3-VL-Embedding-{self.model_size}B",
            name=f"qwen3_vl_{self.model_size}b_embed",
            save_dir=save_dir,
        )

    def __repr__(self):
        return f"This is the multimodal + multilingal Qwen embedding model with {self.model_size}B parameters."


class QwenEmbed(EmbeddingModel):
    VALID_MODEL_SIZES = {
        0: "0.6",
        1: "4",
        2: "8",
    }

    def __init__(self, save_dir: str = "models", size: int = 0):
        if size not in self.VALID_MODEL_SIZES:
            raise ValueError("size must be 0 (0.6B), 1 (4B), or 2 (8B)")

        self.model_size = self.VALID_MODEL_SIZES[size]

        repo_id = f"Qwen/Qwen3-Embedding-{self.model_size}B"

        super().__init__(
            repo_id=repo_id,
            name=f"qwen3_{self.model_size}b_embed",
            save_dir=save_dir,
        )

    def __repr__(self):
        return f"Qwen3 {self.model_size}B parameters multilingual embedding model."


class PerplexityEmbed(EmbeddingModel):
    def __init__(self, save_dir="models", use_small=True):
        self.model_size = 0.6 if use_small else 4
        super().__init__(
            repo_id=f"perplexity-ai/pplx-embed-v1-{self.model_size}b",
            name=f"pplx_{self.model_size}b_embed",
            save_dir=save_dir,
        )

    def __repr__(self):
        return f"This is the {self.model_size}B parameter Perplexity multilingual embedding model."

    def prepare_model(self):
        logger.info('Downloading model')
        self.download()
        logger.info('Loading model')
        self.model = SentenceTransformer(self._download_dir(), trust_remote_code=True)
    
    def embed(self, texts):
        return  self.model.encode(texts)


class PerplexityContextEmbed(EmbeddingModel):
    def __init__(self, save_dir="models", use_small=True):
        self.model_size = 0.6 if use_small else 4
        super().__init__(
            repo_id=f"perplexity-ai/pplx-embed-context-v1-{self.model_size}b",
            name=f"pplx_context_{self.model_size}b_embed",
            save_dir=save_dir,
        )

    def __repr__(self):
        return f"This is the {self.model_size}B parameter context aware Perplexity multilingual embedding model."



class GemmaEmbed(EmbeddingModel):
    def __init__(self, save_dir="models"):
        super().__init__(
            repo_id="google/embeddinggemma-300m", name="gemma_embed", save_dir=save_dir
        )

    def __repr__(self):
        return "Gemma embed is a 300M parameter multilingual embedding model."


class GraniteMultilingualEmbed(EmbeddingModel):
    def __init__(self, save_dir="models"):
        super().__init__(
            repo_id="ibm-granite/granite-embedding-107m-multilingual",
            name="granite_embed",
            save_dir=save_dir,
        )

    def __repr__(self):
        return "Granite is a 107M parameter tiny multilingual embedding model."


if __name__ == "__main__":
    a = GraniteMultilingualEmbed()
    print(a)

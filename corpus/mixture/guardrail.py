from corpus.mixture.guardrail import GuardrailModel


class GPTGuard(GuardrailModel):
    def __init__(self, save_dir="models", use_small=True):
        self.model_size = 20 if use_small else 120
        super().__init__(
            repo_id=f"openai/gpt-oss-safeguard-{self.model_size}b",
            name=f"gptoss_{self.model_size}b_guardrail",
            save_dir=save_dir,
        )


class PerplexityGuard(GuardrailModel):
    def __init__(self, save_dir="models"):
        super().__init__(
            repo_id="perplexity-ai/browsesafe",
            name="pplx_guardrail",
            save_dir=save_dir,
        )

    def __repr__(self):
        return "This model provides guardrail against prompt injection."

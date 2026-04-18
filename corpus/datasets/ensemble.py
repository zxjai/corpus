import io
import json
from pathlib import Path
from urllib.parse import quote, urljoin

import zstandard as zstd
from loguru import logger
from rich.console import Console

from corpus.datasets.base import DataFileExtension, HuggingFaceDataset
from corpus.datasets.constants import DOLMA_BASE_URL, DOLMA_METADATA
from corpus.tools.download import download_single_shard


class NemotronCC(HuggingFaceDataset):
    def __init__(self, save_dir="dataset"):
        super().__init__(
            repo_id="nvidia/Nemotron-CC-v2",
            dataset_name="nemotron_cc_v2",
            save_dir=save_dir,
            data_file_extension=DataFileExtension.PARQUET,
            # include_dir can be modified
            include_dir=["High-Quality-Synthetic", "High-Quality"],
        )


class Dolma:
    def __init__(self, save_dir="dataset"):
        self.repo_id = "allenai/dolma"
        self.dataset_name = "dolma_v7"
        self.version = 7
        self.save_dir = Path(save_dir) / self.dataset_name
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def composition(self) -> None:
        logger.info("Dolma data composition")
        for s in DOLMA_METADATA.keys():
            logger.info(s)

    def get_subset_urls(self, subset_name) -> list[str]:
        num_shards = DOLMA_METADATA[subset_name]["num_shards"]
        urls = []
        for i in range(num_shards):
            url = self._form_url(subset_name=subset_name, shard=i)
            urls.append(url)
        return urls

    def _form_url(self, subset_name: str, shard: int) -> str:
        """
        Shards are zero index, subset_name are the keys of DOLMA_METADATA
        """
        record = DOLMA_METADATA[subset_name]
        folder = record["folder"]
        prefix = record["file_prefix"]
        total_shards = record["num_shards"]
        assert 0 <= shard < total_shards, "In valid index: shard index are zero index"
        path = f"{folder}/{prefix}-{shard:04d}.json.gz"
        url = urljoin(DOLMA_BASE_URL, quote(path))
        return url

    def download_subset(self, subset_name, max_shards=1):
        folder = DOLMA_METADATA[subset_name]["folder"]
        urls = self.get_subset_urls(subset_name)
        save_dir = self.save_dir / folder
        for i, url in enumerate(urls[:max_shards]):
            download_single_shard(
                url=url,
                save_dir=save_dir,
                filename=f"shard_{i:04d}-of-{len(urls)}.json.gz",
                description=f"Downloading {subset_name} shard {i} of {max_shards} shards.",
            )


class DCLM(HuggingFaceDataset):
    def __init__(self, save_dir="dataset"):
        super().__init__(
            repo_id="mlfoundations/dclm-baseline-1.0",
            dataset_name="dclm",
            save_dir=save_dir,
            data_file_extension=DataFileExtension.JSONL_ZST,
        )

    def inspect(self):
        console = Console()
        sample_path = self.sample_local_file()
        with open(sample_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                jsonl = text_stream.readline()
                record = json.loads(jsonl)
                console.print_json(json.dumps(record, indent=4))

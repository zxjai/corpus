from pathlib import Path

from corpus.datasets.base import DataFileExtension, HuggingFaceDataset


class FinePhrase(HuggingFaceDataset):
    def __init__(self, save_dir: Path | str = "dataset"):
        super().__init__(
            repo_id="HuggingFaceFW/finephrase",
            dataset_name="finephrase",
            save_dir=save_dir,
            data_file_extension=DataFileExtension.PARQUET,
            include_dir=["math"],  # many other subsets exists
        )

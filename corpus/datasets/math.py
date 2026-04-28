from corpus.datasets.base import HuggingFaceDataset, DataFileExtension
from pathlib import Path


class MathNet(HuggingFaceDataset):
    """
    Note: this is a multimodal dataset with text and jpg diagrams.
    """

    def __init__(self, save_dir: Path | str = "dataset"):
        super().__init__(
            repo_id="ShadenA/MathNet",
            dataset_name="math_net",
            save_dir=save_dir,
            data_file_extension=DataFileExtension.PARQUET,
        )

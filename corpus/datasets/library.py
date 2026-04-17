import gzip
import json
from pathlib import Path

import pyarrow.csv as pv
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from loguru import logger
from rich.console import Console

from corpus.datasets.base import DataFileExtension, HuggingFaceDataset


class InstitutionalBooks(HuggingFaceDataset):
    def __init__(self, save_dir: Path | str = 'dataset'):
        super().__init__(
            repo_id='institutional/institutional-books-1.0',
            dataset_name='institutional',
            save_dir=save_dir,
            data_file_extension=DataFileExtension.PARQUET)

    def inspect(self) -> None:
        p = self._sample_data_file()
        f = pq.ParquetFile(p)

        record = (
            pq
            .read_table(p)
            .slice(0, 1)
            .to_pandas()
            .iloc[0]
            .to_dict()
        )
        self.inspect_dict(record)

class ProjectGutenberg(HuggingFaceDataset):
    def __init__(self, save_dir: Path | str = 'dataset'):
        super().__init__(
            repo_id='common-pile/project_gutenberg_filtered',
            dataset_name='gutenberg',
            save_dir=save_dir,
            data_file_extension=DataFileExtension.JSON_GZ)

    def inspect(self) -> None:
        path = self._sample_data_file()

        with gzip.open(path, "rt", encoding="utf-8") as f:
            record = json.loads(f.readline())
            self.inspect_dict(record)

class Poems(HuggingFaceDataset):
    def __init__(self,  save_dir: Path | str = 'dataset'):
        super().__init__(
            repo_id='suayptalha/Poetry-Foundation-Poems',
            dataset_name='poems',
            save_dir=save_dir,
            data_file_extension=DataFileExtension.CSV
        )
    
    def inspect(self) -> None:
        p = self._sample_data_file()

        parse_options = pv.ParseOptions(newlines_in_values=True)
        table = pv.read_csv(p, parse_options=parse_options)
        if "" in table.column_names:
            table = table.drop([""])
        record = table.slice(0, 1).to_pylist()[0]
        self.inspect_dict(record)



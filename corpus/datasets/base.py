from __future__ import annotations

import json
from abc import ABC
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from loguru import logger
from rich.console import Console
from rich.table import Table


class DataFileExtension(Enum):
    PARQUET = '.parquet'
    JSON = '.json'
    JSON_GZ = '.json.gz'
    JSONL = '.jsonl'


class HuggingFaceDataset(ABC):
    def __init__(self, repo_id: str, dataset_name: str, save_dir: Path | str, data_file_extension: DataFileExtension):
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.dataset_dir = Path(save_dir) / dataset_name
        self.data_file_extension = data_file_extension

    def process(self):
        """Extract raw files to get processed files."""
        processed_dir = self._processed_dir() # path
        return NotImplemented

    def ls(self, log: bool=True, show_num: int=2) -> list[str]:

        data_files = [
            f for f in list_repo_files(repo_id=self.repo_id, repo_type="dataset") 
            if f.endswith(self.data_file_extension.value)
        ]

        if log:
            logger.info(f'Found {len(data_files)} {self.data_file_extension.value} files')
            for f in data_files[:show_num]:
                logger.info(f)
            logger.info('...')
            for f in data_files[-show_num:]:
                logger.info(f)

        return data_files

    def download_single_file(self, filename: Optional[str]=None) -> None:
        if filename is None:
            files = self.ls(log=False)
            assert len(files), f'No data files found. Datafile extension provided {self.data_file_extension.value}'
            filename = files[0]

        hf_hub_download(
            cache_dir=self._cache_dir(), 
            local_dir=self._download_dir(),
            repo_type="dataset", 
            repo_id=self.repo_id,
            filename=filename
        )

    def download_bulk(self, max_files: Optional[int]=4, max_workers: int=16, log: bool = False):
        if max_files is not None:
            assert max_files > 0
            allow_patterns = self.ls(log=log)[:max_files]
        else:
            allow_patterns=["*"]

        snapshot_download(
            cache_dir=self._cache_dir(), 
            local_dir=self._download_dir(),
            repo_type="dataset", 
            repo_id=self.repo_id, 
            allow_patterns=allow_patterns,
            max_workers=max_workers
        )
    
    def _cache_dir(self) -> Path:
        p = self.dataset_dir / '.cache'
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _download_dir(self) -> Path:
        p = self.dataset_dir / 'raw'
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _processed_dir(self) -> Path:
        p = self.dataset_dir / 'processed'
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sample_data_file(self) -> Optional[str]:
        dd = self._download_dir()
        ext = self.data_file_extension.value
        paths = sorted(dd.rglob(f'*{ext}'))
        if not paths:
            raise FileNotFoundError(f'No *{ext} file found in {dd}')
        return str(paths[0])

    def _keys_tree(self, obj):
        if isinstance(obj, dict):
            return {k: self._keys_tree(v) for k, v in obj.items()}
        return type(obj).__name__


    def _print_keys(self, sample_dict) -> None:
        console = Console()
        table = Table(title="Dataset Keys and Types")
        table.add_column("Key", style="cyan")
        table.add_column("Type", style="green")

        for k, v in sorted(sample_dict.items()):
            t = type(v).__name__ 
            if t not in ['str', 'int', 'bool']:
                t = f'{t} ! '
            table.add_row(str(k), t)

        console.print(table)

    def _print_nested_dict(self, sample_dict):
        tree = self._keys_tree(sample_dict)
        console = Console()
        console.print_json(json.dumps(tree, indent=4))

    def inspect_dict(self, sample_dict):
        self._print_keys(sample_dict)
        self._print_nested_dict(sample_dict)


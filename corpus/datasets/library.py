import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from loguru import logger
from rich.console import Console


class InstitutionalBooks:
    def __init__(self):
        self.tag = 'institutional/institutional-books-1.0'
    
    def ls(self, log=True):
        SHOW_NUM = 2

        parquet_files = [
            f for f in list_repo_files(repo_id=self.tag, repo_type="dataset") 
            if f.endswith(".parquet")
        ]

        if log:
            for f in parquet_files[:SHOW_NUM]:
                logger.info(f)
            logger.info('...')
            for f in parquet_files[-SHOW_NUM:]:
                logger.info(f)
            logger.info(f'Found {len(parquet_files)} parquet files')

        return parquet_files
    
    def download_single_file(self, filename='data/train-00000-of-09831.parquet'):
        hf_hub_download(
            cache_dir='dataset/cache', 
            local_dir='dataset/institution_books',
            repo_type="dataset", 
            repo_id=self.tag,
            filename=filename
        )

    def download_bulk(self, num_files=32, max_workers=16):
        include_files = self.ls(log=False)[:32]
        snapshot_download(
            cache_dir='dataset/cache', 
            local_dir='dataset/institution_books',
            repo_type="dataset", 
            repo_id=self.tag, 
            allow_patterns=include_files,
            max_workers=max_workers
        )
    

    def inspect_parquet(self):
        pq_paths = Path('dataset/institution_books').rglob('*.parquet')
        p = next(pq_paths)
        f = pq.ParquetFile(p)

        row = pq.read_table(p).slice(0, 1).to_pandas().iloc[0]

        def keys_tree(obj):
            if isinstance(obj, dict):
                return {k: keys_tree(v) for k, v in obj.items()}
            return type(obj).__name__

        tree = keys_tree(row.to_dict())

        console = Console()
        logger.info('Displaying keys')
        console.print_json(json.dumps(tree, indent=4))
        logger.info(f'Num rows {f.metadata.num_rows} found in {p}')
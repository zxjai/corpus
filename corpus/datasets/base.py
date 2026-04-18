from __future__ import annotations

import bz2
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import fsspec
import requests
from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.hf_api import RepoFolder
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


class DataFileExtension(Enum):
    CSV = ".csv"
    JSON = ".json"
    JSON_GZ = ".json.gz"
    JSONL_ZST = ".jsonl.zst"
    JSONL = ".jsonl"
    PARQUET = ".parquet"
    XML_GZ = ".xml.bz2"


class WikipediaDataset:
    def __init__(
        self,
        base_url,
        name="wiki",
        save_dir="dataset",
        extension=DataFileExtension.XML_GZ,
    ):
        self.base_url = base_url
        self.save_dir = Path(save_dir) / name
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.extension = extension

    def get_shards_paths(self, timeout=30) -> list[str]:
        res = requests.get(self.base_url, timeout=timeout)
        res.raise_for_status()
        pat = r'href="(enwiki\w*-[\w.-]+\.xml\.bz2)"'
        return re.findall(pat, res.text)

    def download_single_shard(
        self, shard_path: Optional[str] = None, description: str = None
    ):

        if shard_path is None:
            shard_path = sorted(self.get_shards_paths())[0]

        url = f"{self.base_url}/{shard_path}"
        save_path = self._download_dir() / shard_path

        lock_path = save_path.with_suffix(save_path.suffix + ".lock")
        part_path = save_path.with_suffix(save_path.suffix + ".part")

        with FileLock(str(lock_path)):
            if save_path.exists() and save_path.stat().st_size > 0:
                logger.info(f"{save_path} exists")
                return save_path

            offset = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}

            with requests.get(url, stream=True, headers=headers) as r:
                if offset and r.status_code == 416:
                    logger.info(f"{shard_path} fully downloaded, skipping")
                    os.replace(part_path, save_path)
                    return save_path
                if offset and r.status_code == 200:
                    logger.warning(
                        f"Server ignored range for {shard_path}, re-downloading"
                    )
                    part_path.unlink(missing_ok=True)
                    offset = 0

                r.raise_for_status()
                content_len = int(r.headers.get("content-length", 0))
                total_len = offset + content_len if content_len else None
                mode = "ab" if offset else "wb"

                desc = description if description else shard_path
                with (
                    open(part_path, mode) as f,
                    tqdm(
                        initial=offset,
                        total=total_len,
                        unit="B",
                        unit_scale=True,
                        desc=desc,
                    ) as progress_bar,
                ):
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            if total_len is not None and part_path.stat().st_size != total_len:
                raise IOError(f"Download for {shard_path} is incomplete")
            os.replace(part_path, save_path)
            logger.info(f"Downloaded {shard_path}")

        return save_path

    def download_bulk(self):

        max_workers = 3  # Wikipedia's concurrent requests is 3

        files = self.get_shards_paths()
        assert len(files), "No files found"
        logger.info(f"Donwloading {len(files)} shards")
        downloaded = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_shard = {
                executor.submit(
                    self.download_single_shard, shard, f"file-{i + 1}-of-{len(files)}"
                ): shard
                for i, shard in enumerate(files)
            }
            for future in as_completed(future_to_shard):
                shard = future_to_shard[future]
                try:
                    downloaded.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to download {shard}: {e}")

        logger.info(f"Downloaded {len(downloaded)}/{len(files)} shards")

        return downloaded

    def inspect(self):
        paths = sorted(self._download_dir().rglob(f"*{self.extension.value}"))
        assert len(paths), f"No *{self.extension.value} files found"
        p = paths[0]
        console = Console()

        with bz2.open(p, "rb") as f:
            NS = "{http://www.mediawiki.org/xml/export-0.11/}"
            i = 0
            for _, record in ET.iterparse(f, events=("end",)):
                i += 1

                if record.tag != f"{NS}page":
                    continue

                namespace = record.findtext(f"{NS}ns")  # main article namespace is 0
                if namespace != "0":
                    record.clear()
                    continue

                if record.find(f"{NS}redirect") is not None:
                    record.clear()
                    continue

                revision = record.find(f"{NS}revision")
                text_el = revision.find(f"{NS}text")
                text = text_el.text if text_el is not None else None

                if text and text.lstrip().upper().startswith("#REDIRECT"):
                    record.clear()
                    continue

                # xml_text = ET.tostring(record, encoding="unicode")

                page = {
                    "title": record.findtext(f"{NS}title"),
                    "text": text,
                    # 'page_id': record.findtext(f'{NS}id'),
                    # 'timestamp': revision.findtext(f'{NS}timestamp'),
                    # 'model': revision.findtext(f'{NS}model'),
                    # 'bytes': text_el.get('bytes') if text_el is not None else None,
                }

                # EARLIEST_DATE_CUTOFF = '2025-01-01T00:00:00Z'
                # if page['timestamp'] and page['timestamp'] < EARLIEST_DATE_CUTOFF:
                #     console.print_json(json.dumps(page, indent=4))

                if i % 1000 == 0:
                    logger.info(f"Seen {i} pages")
                    console.print_json(json.dumps(page, indent=4))
                    break

                # logger.info(xml_text)
                record.clear()

    def remote_open(self, shard_path: Optional[str] = None):
        """
        Shard path is relative to base_url
        Example
            enwiki-2026-04-01-p23970571p28987923.xml.bz2
        """
        if shard_path is None:
            paths = self.get_shards_paths()
            assert paths
            shard_path = paths[0]

        bz2_url = f"{self.base_url}/{shard_path}"
        with fsspec.open(bz2_url, "r", compression="bz2") as f:
            # do work
            logger.info(f.readline())

    def _download_dir(self) -> Path:
        p = self.save_dir / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _processed_dir(self) -> Path:
        p = self.save_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p


class HuggingFaceDataset:
    def __init__(
        self,
        repo_id: str,
        dataset_name: str,
        save_dir: Path | str,
        data_file_extension: DataFileExtension,
        include_dir: Optional[list[str]] = None,
    ):
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.dataset_dir = Path(save_dir) / dataset_name
        self.data_file_extension = data_file_extension
        if include_dir:
            self.include_dir = include_dir

    def process(self):
        """Extract raw files to get processed files."""
        # processed_dir = self._processed_dir()  # path
        return NotImplemented

    def ls(self, log: bool = True, show_num: int = 2, recursive=True) -> list[str]:

        # normalize search patterns
        suffix = "/**" if recursive else "/*"
        patterns = (
            [p if "*" in p else f"{p.rstrip('/')}{suffix}" for p in self.include_dir]
            if self.include_dir
            else None
        )

        data_files = [
            f
            for f in list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            if f.endswith(self.data_file_extension.value)
            and (patterns is None or any(fnmatch(f, pat) for pat in patterns))
        ]

        if log:
            logger.info(
                f"Found {len(data_files)} {self.data_file_extension.value} files"
            )

            print(len(data_files), show_num)
            if len(data_files) <= 2 * show_num:
                for f in data_files:
                    logger.info(f)
            else:
                for f in data_files[:show_num]:
                    logger.info(f)
                logger.info("...")
                for f in data_files[-show_num:]:
                    logger.info(f)
        return data_files

    def ls_all(self, folders_only=True, recursive=True):
        results = []
        i = 0
        api = HfApi()
        for item in api.list_repo_tree(
            self.repo_id, repo_type="dataset", recursive=recursive
        ):
            kind = "directory" if isinstance(item, RepoFolder) else "file"
            if kind == "directory":
                i += 1
                logger.info(f"item {i} | {kind} | {item.path}")
                results.append(item.path)
            if not folders_only and kind == "file":
                i += 1
                logger.info(f"item {i} | {kind} | {item.path}")
                results.append(item.path)

        logger.info(f"Found {len(results)} items.")
        return results

    def download_single_file(self, filename: Optional[str] = None) -> None:
        if filename is None:
            files = self.ls(log=False)
            assert len(files), (
                f"No data files found. Datafile extension provided {self.data_file_extension.value}"
            )
            filename = files[0]

        hf_hub_download(
            cache_dir=self._cache_dir(),
            local_dir=self._download_dir(),
            repo_type="dataset",
            repo_id=self.repo_id,
            filename=filename,
        )

    def download_bulk(
        self, max_files: Optional[int] = 4, max_workers: int = 16, log: bool = False
    ):
        if max_files is not None:
            assert max_files > 0
            allow_patterns = self.ls(log=log)[:max_files]
        else:
            allow_patterns = ["*"]

        snapshot_download(
            cache_dir=self._cache_dir(),
            local_dir=self._download_dir(),
            repo_type="dataset",
            repo_id=self.repo_id,
            allow_patterns=allow_patterns,
            max_workers=max_workers,
        )

    def _cache_dir(self) -> Path:
        p = self.dataset_dir / ".cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _download_dir(self) -> Path:
        p = self.dataset_dir / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _processed_dir(self) -> Path:
        p = self.dataset_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sample_data_file(self) -> Optional[str]:
        dd = self._download_dir()
        ext = self.data_file_extension.value
        paths = sorted(dd.rglob(f"*{ext}"))
        if not paths:
            raise FileNotFoundError(f"No *{ext} file found in {dd}")
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
            if t not in ["str", "int", "bool"]:
                t = f"{t} ! "
            table.add_row(str(k), t)

        console.print(table)

    def _print_nested_dict(self, sample_dict):
        tree = self._keys_tree(sample_dict)
        console = Console()
        console.print_json(json.dumps(tree, indent=4))

    def inspect_dict(self, sample_dict):
        self._print_keys(sample_dict)
        self._print_nested_dict(sample_dict)

    def sample_local_file(self):
        search_pattern = "*" + self.data_file_extension.value
        paths = sorted(self._download_dir().rglob(search_pattern))
        assert len(paths), f"No files found with extension {search_pattern}"
        return paths[0]

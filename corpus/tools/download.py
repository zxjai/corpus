import os
from pathlib import Path

import requests
from filelock import FileLock
from loguru import logger
from tqdm import tqdm


def download_single_shard(
    url: str,
    save_dir: str | Path,
    filename: str,
    description: str | None = None,
    timeout: int = 60,
) -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path / filename

    lock_path = save_path.with_suffix(save_path.suffix + ".lock")
    part_path = save_path.with_suffix(save_path.suffix + ".part")

    with FileLock(str(lock_path)):
        if save_path.exists() and save_path.stat().st_size > 0:
            logger.info(f"{save_path} exists")
            return save_path

        offset = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
            if offset and r.status_code == 416:
                logger.info(f"{filename} fully downloaded, skipping")
                os.replace(part_path, save_path)
                return save_path
            if offset and r.status_code == 200:
                logger.warning(f"Server ignored range for {filename}, re-downloading")
                part_path.unlink(missing_ok=True)
                offset = 0

            r.raise_for_status()
            content_len = int(r.headers.get("content-length", 0))
            if not content_len:
                logger.warning(
                    f"No content-length for {filename}; progress bar will lack total"
                )
            total_len = offset + content_len if content_len else None
            mode = "ab" if offset else "wb"

            desc = description if description is not None else filename
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
            raise IOError(f"Download for {filename} is incomplete")
        os.replace(part_path, save_path)
        logger.info(f"Downloaded {filename}")

    return save_path

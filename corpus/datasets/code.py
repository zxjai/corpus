import gzip
import json
import urllib.request
from collections import Counter
from pathlib import Path

from loguru import logger
from tqdm import tqdm


class GitHubDiscovery:
    def __init__(self, save_dir="dataset"):
        self.save_dir = Path(save_dir) / "github_archive"
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def ls_archive_files(self):
        return self.save_dir.rglob("*.json.gz")

    def download_hour(self, hour: int = 0, date="2026-04-16"):

        BASE_URL = "https://data.gharchive.org/"
        path = f"{date}-{hour}.json.gz"

        url = BASE_URL + path
        dest = self.save_dir / path

        if dest.exists():
            logger.info(f"Already downloaded {dest.name}")
            return str(dest)

        logger.info(f"Downloading {url}")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; gharchive-downloader/1.0)"
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            while chunk := resp.read(1 << 16):
                f.write(chunk)
        logger.info(f"Downloaded {url}")
        return str(dest)

    def download_day(self, date="2026-04-16"):
        for h in range(24):
            self.download_hour(hour=h, date=date)

    def star_frequency(self):
        fail_count = 0
        stars = Counter()
        # console = Console()
        files = list(self.ls_archive_files())
        assert files, f"No .json.gz found in {self.save_dir}"

        for path in tqdm(files, desc="Computing star freqency"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    # console.print_json(json.dumps(record, indent=4))
                    try:
                        event = record["type"]
                        repo_name = record["repo"]["name"]
                        if event == "WatchEvent":
                            stars[repo_name] += 1
                    except Exception:
                        fail_count += 1

        stars = stars.most_common()
        stars = dict(stars)

        f_path = self._stats_path() / "star_frequency.json"
        with open(f_path, "w") as g:
            g.write(json.dumps(stars, indent=4))

        logger.info(f"Saved at {f_path} | fail count {fail_count}")

    def _stats_path(self) -> Path:
        p = self.save_dir / "statistics"
        p.mkdir(parents=True, exist_ok=True)
        return p

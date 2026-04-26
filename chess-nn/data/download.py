import argparse
import pathlib
import time
import zipfile

import requests
import zstandard as zstd


SMOKE_TEST_URLS = [
    "https://database.nikonoel.fr/lichess_elite_2024-01.pgn.zst",
]

FULL_RUN_URLS = [
    "https://database.nikonoel.fr/lichess_elite_2024-01.pgn.zst",
    "https://database.nikonoel.fr/lichess_elite_2024-02.pgn.zst",
    "https://database.nikonoel.fr/lichess_elite_2024-03.pgn.zst",
    "https://database.nikonoel.fr/lichess_elite_2024-04.pgn.zst",
    "https://database.nikonoel.fr/lichess_elite_2024-05.pgn.zst",
    "https://database.nikonoel.fr/lichess_elite_2024-06.pgn.zst",
]


def local_storage_root() -> pathlib.Path:
    return pathlib.Path.home() / "chess-nn-data"


def format_size(num_bytes: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(suffixes) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f}{suffixes[idx]}"


def download_with_retries(url: str, dst: pathlib.Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with dst.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100.0 / total
                            print(f"\rDownloading {dst.name}: {format_size(downloaded)} / {format_size(total)} ({pct:.1f}%)", end="")
                        else:
                            print(f"\rDownloading {dst.name}: {format_size(downloaded)}", end="")
                print()
            return
        except Exception as exc:
            print(f"Download attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt == retries:
                raise
            time.sleep(2)


def is_zstd_file(path: pathlib.Path) -> bool:
    with path.open("rb") as f:
        magic = f.read(4)
    return magic == b"\x28\xb5\x2f\xfd"


def decompress_zst(src: pathlib.Path, dst: pathlib.Path) -> None:
    compressed_size = src.stat().st_size
    dctx = zstd.ZstdDecompressor()
    written = 0
    with src.open("rb") as infile, dst.open("wb") as outfile:
        with dctx.stream_reader(infile) as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                outfile.write(chunk)
                written += len(chunk)
                ratio = (infile.tell() / compressed_size * 100.0) if compressed_size > 0 else 0.0
                print(f"\rDecompressing {src.name}: in={format_size(infile.tell())} out={format_size(written)} ({ratio:.1f}% input read)", end="")
    print()


def decompress_zip(src: pathlib.Path, dst: pathlib.Path) -> None:
    with zipfile.ZipFile(src, "r") as zf:
        pgn_members = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not pgn_members:
            raise ValueError(f"No .pgn file found in zip: {src}")
        member = pgn_members[0]
        info = zf.getinfo(member)
        with zf.open(member) as infile, dst.open("wb") as outfile:
            copied = 0
            while True:
                chunk = infile.read(1024 * 1024)
                if not chunk:
                    break
                outfile.write(chunk)
                copied += len(chunk)
                total = info.file_size
                pct = (copied * 100.0 / total) if total > 0 else 0.0
                print(f"\rDecompressing {src.name}: {format_size(copied)} / {format_size(total)} ({pct:.1f}%)", end="")
    print()


def validate_pgn_file(path: pathlib.Path) -> None:
    with path.open("rb") as f:
        first = f.read(4096)
    text = first.decode("utf-8")
    normalized = text.lstrip("\ufeff\r\n\t ")
    if "[Event " not in normalized:
        raise ValueError(f"{path} does not appear to be valid PGN (missing [Event header at start)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and decompress Lichess Elite PGN files.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    urls = SMOKE_TEST_URLS if args.mode == "smoke" else FULL_RUN_URLS
    repo_raw_dir = pathlib.Path(__file__).resolve().parent / "raw"
    repo_raw_dir.mkdir(parents=True, exist_ok=True)
    storage_raw_dir = local_storage_root() / "raw"
    storage_raw_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        zst_path = storage_raw_dir / filename
        pgn_path = storage_raw_dir / filename.replace(".pgn.zst", ".pgn")
        using_zip_fallback = False

        if zst_path.exists():
            print(f"Skipping download (exists): {zst_path.name}")
        else:
            print(f"Downloading {filename} ...")
            download_with_retries(url, zst_path)

        if not is_zstd_file(zst_path):
            print(f"Downloaded file is not Zstandard: {zst_path.name}")
            zip_url = url.replace(".pgn.zst", ".zip")
            zip_path = storage_raw_dir / filename.replace(".pgn.zst", ".zip")
            if zip_path.exists():
                print(f"Using existing ZIP fallback: {zip_path.name}")
            else:
                print(f"Falling back to ZIP source: {zip_url}")
                download_with_retries(zip_url, zip_path)
            using_zip_fallback = True

        if pgn_path.exists():
            print(f"Skipping decompression (exists): {pgn_path.name}")
        else:
            if using_zip_fallback:
                zip_path = storage_raw_dir / filename.replace(".pgn.zst", ".zip")
                print(f"Decompressing {zip_path.name} -> {pgn_path.name}")
                decompress_zip(zip_path, pgn_path)
            else:
                print(f"Decompressing {zst_path.name} -> {pgn_path.name}")
                decompress_zst(zst_path, pgn_path)

        validate_pgn_file(pgn_path)
        print(f"OK: {zst_path.name} ({format_size(zst_path.stat().st_size)})")
        zip_path = storage_raw_dir / filename.replace(".pgn.zst", ".zip")
        if zip_path.exists():
            print(f"OK: {zip_path.name} ({format_size(zip_path.stat().st_size)})")
        print(f"OK: {pgn_path.name} ({format_size(pgn_path.stat().st_size)})")

        repo_zst = repo_raw_dir / filename
        repo_pgn = repo_raw_dir / filename.replace(".pgn.zst", ".pgn")
        repo_zip = repo_raw_dir / filename.replace(".pgn.zst", ".zip")
        repo_zst.write_text(str(zst_path), encoding="utf-8")
        repo_pgn.write_text(str(pgn_path), encoding="utf-8")
        if zip_path.exists():
            repo_zip.write_text(str(zip_path), encoding="utf-8")


if __name__ == "__main__":
    main()

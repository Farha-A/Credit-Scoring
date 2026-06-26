"""Utilities for downloading and decompressing the raw LendingClub dataset from Google Drive."""

import gzip
import shutil

import gdown


def download_data(folder_id: str) -> None:
    """Download all content from a Google Drive folder.

    Args:
        folder_id: The Google Drive folder ID to download from.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading from folder: {folder_id}")
    gdown.download_folder(url=url, output="./downloaded_folder", quiet=False, use_cookies=False)


def decompress_gz(file_path: str, output_file_name: str) -> None:
    """Decompress a gzip-compressed file.

    Args:
        file_path: Path to the .gz file to decompress.
        output_file_name: Destination path for the decompressed output.
    """
    print(f"Decompressing {file_path} -> {output_file_name}")
    with gzip.open(file_path, "rb") as f_in:
        with open(output_file_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_raw_dataset(
    folder_id: str,
    downloaded_gz_path: str,
    decompressed_csv_name: str,
) -> None:
    """Download and decompress the raw LendingClub dataset.

    Combines :func:`download_data` and :func:`decompress_gz` into a single
    convenience call.

    Args:
        folder_id: Google Drive folder ID that contains the .gz archive.
        downloaded_gz_path: Local path where the .gz file lands after download.
        decompressed_csv_name: Local path for the resulting CSV file.
    """
    download_data(folder_id)
    decompress_gz(downloaded_gz_path, decompressed_csv_name)

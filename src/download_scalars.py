#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:41:50 2025

@author: asayeed
"""

import gdown
from pathlib import Path

def download_from_gdrive(file_id, filename="file.csv"):
    """
    Downloads a file from public Google Drive using its file ID and saves to Scalars/.

    Args:
        file_id (str): Google Drive file ID.
        filename (str): Local filename to save as (inside Scalars/).
    """
    scalar_path = Path(__file__).resolve().parent.parent / "Scalars"
    scalar_path.mkdir(parents=True, exist_ok=True)
    output_path = scalar_path / filename

    if not output_path.is_file() or output_path.stat().st_size == 0:
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            print(f"Downloading {filename} from Google Drive...")
            result = gdown.download(url, str(output_path), quiet=False)
            if result is None or not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("Download did not produce a nonempty file")
        except Exception as e:
            raise RuntimeError(f"Failed to download required input {filename} from {url}") from e
    else:
        print(f"{filename} already exists at Scalars/")

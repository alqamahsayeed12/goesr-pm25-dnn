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

    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            print(f"Downloading {filename} from Google Drive...")
            gdown.download(url, str(output_path), quiet=False)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"{filename} already exists at Scalars/")

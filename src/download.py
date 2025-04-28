#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:17:18 2025

@author: asayeed
"""

"""
Module for downloading GOES PM2.5 NetCDF files from STAR NESDIS public repository.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta

# Function to generate a range of dates between start_date and end_date
def daterange(start_date, end_date):
    """
    Yields each date from start_date to end_date.
    """
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)
        
        
def download_goes(start, end, wd_folder="./downloads"):
    
    """
    Downloads GOES PM2.5 files between start and end dates.

    Args:
        start (str): Start date in 'YYYYMMDD' format.
        end (str): End date in 'YYYYMMDD' format.
        wd_folder (str): Working directory where downloads will be stored.

    Returns:
        List of file paths downloaded.
    """
    
    base_url = "https://www.star.nesdis.noaa.gov/pub/smcd/hzhang/GOES/pm25gwr"
    wd_path = Path(wd_folder).resolve()
    goes_root = wd_path.parent / "GOES"
    goes_root.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    start_date = datetime.strptime(start, "%Y%m%d").date()
    end_date = datetime.strptime(end, "%Y%m%d").date()

    for date in daterange(start_date, end_date):
        url = f"{base_url}/{date.strftime('%Y%m%d')}/"
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.endswith(".nc"):
                    file_url = url + href
                    date_folder = goes_root / date.strftime('%Y%m%d')
                    date_folder.mkdir(parents=True, exist_ok=True)
                    local_path = date_folder / href

                    if not local_path.exists():
                        file_resp = requests.get(file_url, stream=True)
                        with open(local_path, 'wb') as f:
                            for chunk in file_resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        downloaded_files.append(str(local_path))
        except Exception as e:
            print(f"Error accessing {url}: {e}")
    return downloaded_files
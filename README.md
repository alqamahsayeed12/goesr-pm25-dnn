# GOES-R PM2.5 Estimation Pipeline

![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Built with Conda](https://img.shields.io/badge/environment-conda-blue.svg)

A reproducible, modular pipeline to estimate surface-level PM2.5 concentrations from GOES-R satellite AOD and HRRR meteorological data using deep learning models.

---

## Overview

This pipeline enables real-time and retrospective monitoring of fine particulate matter (PM2.5) using high-temporal-resolution GOES-R AOD data combined with meteorological inputs from the High-Resolution Rapid Refresh (HRRR) model. It applies regionally optimized deep neural networks (DNNs) and an ensemble model to generate hourly PM2.5 estimates at 2–5 km spatial resolution.


<img width="1280" height="1050" alt="image" src="https://github.com/user-attachments/assets/89407db5-1e59-4e30-80c7-157559149ed4" />



### Key Features
- Near real-time PM2.5 estimation using GOES ABI AOD
- Integration of HRRR-based meteorological features (RH, temperature, wind speed)
- Spatial smoothing, SZA/SED correction, and quality checks
- Ensemble modeling across 14 regional DNN models
- Outputs georeferenced NetCDF files and PM2.5 visualizations

---

## Associated Publication
This pipeline is based on the methodology published in:

> **Sayeed, A., Gupta, P., Henderson, B., Kondragunta, S., Zhang, H., & Liu, Y.** (2025).  
> *GOES-R PM2.5 Evaluation and Bias Correction: A Deep Learning Approach*.  
> Earth and Space Science, 12(2), e2024EA004012.  
> [https://doi.org/10.1029/2024EA004012](https://doi.org/10.1029/2024EA004012)

Please cite this paper when using this pipeline in your research.

BibTeX Citation:
```bibtex
@article{https://doi.org/10.1029/2024EA004012,
  author = {Sayeed, Alqamah and Gupta, Pawan and Henderson, Barron and Kondragunta, Shobha and Zhang, Hai and Liu, Yang},
  title = {GOES-R PM2.5 Evaluation and Bias Correction: A Deep Learning Approach},
  journal = {Earth and Space Science},
  volume = {12},
  number = {2},
  pages = {e2024EA004012},
  keywords = {AOD, PM2.5, deep learning, GOES-R, remote sensing, ABI},
  doi = {https://doi.org/10.1029/2024EA004012},
  url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024EA004012},
  year = {2025}
}
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/alqamahsayeed12/goesr-pm25-dnn.git
cd goesr-pm25-dnn
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate goes-pm25-dnn
```

### 3. Prepare Required Files
Ensure the following are available inside the `Scalars/` folder:
- `collocated_hrrr_on_goes.csv`
- `hrrr_varaible_list_selected4.csv`
- `max_min.csv`

Ensure trained models are placed under the `Models/` folder.

---

## Running the Pipeline

Download and process GOES data using:
```bash
python main.py --start 20210113 --end 20210113 --plot True
```
Arguments:
- `--start`: Start date (`YYYYMMDD`)
- `--end`: End date (`YYYYMMDD`)
- `--plot`: Optional (`True` or `False`) to generate PM2.5 plots

---

## Expected Outputs

| Folder        | Description                                                  |
|:--------------|:--------------------------------------------------------------|
| `OUT_NC/`     | NetCDF files with predicted PM2.5 for each time step           |
| `PLOTS/`      | PNG visualizations comparing GOES-derived and predicted PM2.5 |
| `GOES/`       | Downloaded raw GOES NetCDF data                               |

---

## Folder Structure
```
goes-pm25-dnn/
├── src/
│   ├── config.py
│   ├── download.py
│   ├── download_scalars.py
│   ├── preprocess.py
│   ├── pipeline.py
│   ├── utils.py
├── main.py
├── environment.yml
├── OUT_NC/
├── PLOTS/
├── GOES/
├── Models/
├── Scalars/
└── README.md
```

---

## Dependencies

Install all dependencies automatically:
```bash
conda env create -f environment.yml
```
Key libraries used:
- `xarray`, `pandas`, `numpy`, `matplotlib`, `keras`, `tensorflow`
- `cartopy`, `requests`, `beautifulsoup4`, `tqdm`, `argparse`

---
## Contact
**Project Lead: Pawan Gupta**  
Co-Lead: AERONET Program, NASA  
Email: pawan.gupta@nasa.gov  
ORCID: [0000-0002-0979-472X](https://orcid.org/0000-0002-0979-472X)

**Developer: Alqamah Sayeed, Ph.D.**  
Lead – Air Quality & Health, SERVIR Science Coordination Office, NASA  
Email: alqamah.sayeed@nasa.gov  
ORCID: [0000-0001-6898-8148](https://orcid.org/0000-0001-6898-8148)

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License** (CC BY-NC-ND 4.0).  

See full license terms at [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---

# GOES-R PM2.5 Estimation Pipeline

![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Built with Conda](https://img.shields.io/badge/environment-conda-blue.svg)

A reproducible, modular pipeline to estimate surface-level PM2.5 concentrations from GOES-R satellite AOD and HRRR meteorological data using deep learning models.

---

## Overview

This pipeline enables real-time and retrospective monitoring of fine particulate matter (PM2.5) using high-temporal-resolution GOES-R AOD data combined with meteorological inputs from the High-Resolution Rapid Refresh (HRRR) model. It applies regionally optimized deep neural networks (DNNs) and an ensemble model to generate hourly PM2.5 estimates at 2–5 km spatial resolution.

### Key Features
- Near real-time PM2.5 estimation using GOES ABI AOD (5-min temporal resolution)
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
git clone https://github.com/YOUR_USERNAME/goes-pm25-pipeline.git
cd goes-pm25-pipeline
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate goes-pm25-pipeline
```

### 3. Download GOES Data
Edit `download_goes.py` with your desired start and end dates.
Then run:
```bash
python download_goes.py
```
This will create `GOES/YYYYMMDD/` folders and generate `downloaded_dates.txt`.

### 4. Run the Processing Pipeline
```bash
python main.py
```
This will apply the regional and ensemble DNN models and output hourly results.

---

## Expected Outputs

| Folder        | Description                                                              |
|---------------|---------------------------------------------------------------------------|
| `OUT_NC/`     | NetCDF files with predicted PM2.5 fields for each time step              |
| `PLOTS/`      | PNG maps visualizing surface PM2.5 for GOES timestamps                   |
| `logs/`       | Timestamped logs capturing progress, errors, and runtime diagnostics     |

All outputs are georeferenced using latitude and longitude from the merged GOES grid.

---

## Folder Structure
```
goes-pm25-pipeline/
├── download_goes.py
├── main.py
├── goes_pipeline_core.py
├── config.py
├── utils/
│   ├── solar.py
│   ├── merge.py
│   ├── preprocess.py
│   ├── models.py
├── logs/
├── OUT_NC/
├── PLOTS/
├── run_pipeline.sh
├── environment.yml
├── README.md
├── LICENSE
└── .gitignore
```

---

## Dependencies

All required dependencies can be installed using:
```bash
conda env create -f environment.yml
```
Key packages include:
- `xarray`, `pandas`, `numpy`, `matplotlib`, `keras`, `tensorflow`
- `cartopy`, `requests`, `beautifulsoup4`, `pysolar`

---

## Contact
**Alqamah Sayeed, Ph.D.**  
Lead – Air Quality & Health, SERVIR Science Coordination Office, NASA  
Email: alqamah.sayeed@nasa.gov  
ORCID: [0000-0001-6898-8148](https://orcid.org/0000-0001-6898-8148)

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License** (CC BY-NC-ND 4.0).  

See full license terms at [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).

---

# UAV-LiDAR Carbon Estimation in Post-Hurricane Mangroves

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)]()

Code and data for the paper:

> **Less is more: two UAV-LiDAR metrics outperform complex machine learning algorithms
> for aboveground carbon estimation in hurricane-disturbed mangroves**
> Troche-Souza C. et al. — *submitted to JAG, 2025*

---

## Overview

This repository contains the full processing pipeline to:

1. Clip UAV-LiDAR point clouds (`.las`) to field plot boundaries
2. Classify ground and vegetation returns without external ground control
3. Correct systematic RTK vertical offsets
4. Extract 15 structural LiDAR metrics per plot per season
5. Correlate LiDAR metrics with field-measured aboveground carbon (AGC)
6. Compare six modeling approaches using leave-one-out cross-validation
7. Deploy the optimal model on new flights without field data

**Study site:** Marismas Nacionales, Nayarit, Mexico  
**Sensor:** DJI Zenmuse L1 on Matrice 300 RTK  
**Campaigns:** 4 seasons (dry/wet 2023 and 2024)  
**Plots:** 26 permanent SMMM monitoring plots (20 × 20 m)  
**AGC range:** 1.50 – 71.30 Mg C ha⁻¹  
**Best model:** Linear regression with h_p99 + rumple index (R² = 0.54, RMSE = 11.41 Mg C ha⁻¹)

---

## Key finding

A parsimonious linear model using only two LiDAR metrics —
the 99th percentile of normalized canopy height (**h_p99**) and
the canopy surface rugosity index (**rumple**) — outperformed
Random Forest, Gradient Boosting, and deep neural networks
under leave-one-out cross-validation with n = 48:

See `results/model/ecuacion.txt` for the exact coefficients and
`results/model/metadata.json` for full model metadata.

---

## Repository structure
manglaria-uav-lidar-agc/
├── data/
│   ├── field/
│   │   └── agc_plots.csv          # AGC and disturbance score per plot
│   └── lidar_metrics/
│       └── metricas_lidar_principal.csv  # 15 LiDAR metrics per plot-campaign
├── scripts/
│   ├── 01_clip_y_clasificar.py    # Clip .las by plot + RTK correction + classification
│   ├── 02_extraer_metricas.py     # Extract 15 structural metrics
│   ├── 03_correlacion_agc.py      # Pearson/Spearman correlations + heatmap by season
│   ├── 04_modelo_base.py          # Shared functions for all TF models
│   ├── 04a_modelo_lluvias2024.py  # Model A: wet season 2024 only
│   ├── 04b_modelo_todas_epocas.py # Model B: all seasons, no disturbance feature
│   ├── 04c_modelo_con_disturbio.py# Model C: all seasons + disturbance as feature
│   ├── 05_comparacion_modelos.py  # Full model comparison (6 approaches)
│   ├── 06_modelo_optimo.py        # Final optimal model training and export
│   └── 07_inferencia.py           # Inference on new flights without field data
├── results/
│   ├── figures/                   # Key figures from the paper
│   └── model/
│       ├── ecuacion.txt           # Model equation in plain text
│       └── metadata.json          # Model metadata and LOO metrics
├── environment.yml                # Conda environment
├── LICENSE
└── README.md

---

## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/manglaria-uav-lidar-agc.git
cd manglaria-uav-lidar-agc

# Create and activate the conda environment
conda env create -f environment.yml
conda activate manglaria
```

---

## Quick start

If you only want to reproduce the model comparison from the pre-computed metrics:
```bash
cd scripts
python 05_comparacion_modelos.py
```

To run inference on a new site with your own LiDAR metrics CSV:
```bash
python 07_inferencia.py \
    --metricas /path/to/your/metricas.csv \
    --region   YourSiteName \
    --salida   /path/to/output.csv
```

To run the full pipeline from raw `.las` files, edit the paths at the top
of each script to point to your data directories, then run scripts 01–06 in order.

---

## Data

| File | Description | n rows |
|------|-------------|--------|
| `data/field/agc_plots.csv` | Plot-level AGC (Mg C ha⁻¹) and hurricane disturbance score | 26 |
| `data/lidar_metrics/metricas_lidar_principal.csv` | 15 LiDAR metrics per plot-campaign combination | 48 |

Raw `.las` point clouds are not included due to file size (~172M points per flight).
They are available on request from the corresponding authors.

---

## Dependencies

Main packages (see `environment.yml` for full list):

| Package | Version | Purpose |
|---------|---------|---------|
| laspy | ≥ 2.0 | Read/write .las/.laz files |
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.10 | Spatial interpolation, statistics |
| pandas | ≥ 2.0 | Tabular data |
| geopandas | ≥ 0.13 | Spatial clip by plot polygon |
| shapely | ≥ 2.0 | Geometry operations |
| scikit-learn | ≥ 1.3 | RF, Ridge, Lasso, metrics |
| tensorflow | ≥ 2.13 | Neural network models |
| matplotlib | ≥ 3.7 | Figures |

---

## Citation

If you use this code or data, please cite:
```bibtex
@article{troche2026manglaria,
  title   = {Less is more: two UAV-LiDAR metrics outperform complex machine
             learning algorithms for aboveground carbon estimation in
             hurricane-disturbed mangroves},
  author  = {Troche-Souza, Carlos and others},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The field data in `data/field/agc_plots.csv` derives from the
Mexican Mangrove Monitoring System (SMMM, CONABIO) and is shared
here for reproducibility under the same CC-BY terms as the associated publication.

---

## Contact

Carlos Troche-Souza — ctroche@conabio.gob.mx  

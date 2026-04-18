# 🌫️ Beijing Air Quality Analyser
### CMP7005 PRAC1 — Cardiff Metropolitan University · School of Technologies

---

## Overview

A full end-to-end air quality analysis platform for the **Beijing Multi-Site Air Quality Dataset (2013–2017)**. The project combines an interactive **Streamlit dashboard** with a complete **inline analysis pipeline** covering data loading, preprocessing, exploratory data analysis, and machine learning modelling — all runnable from a single Google Colab notebook.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Sections](#notebook-sections)
- [Streamlit App Pages](#streamlit-app-pages)
- [Machine Learning Models](#machine-learning-models)
- [Key Findings](#key-findings)
- [Dependencies](#dependencies)
- [References](#references)

---

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI Machine Learning Repository — Beijing Multi-Site Air Quality Data](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) |
| Stations | 12 monitoring stations across Beijing |
| Period | March 2013 – February 2017 |
| Records | ~35,064 hourly records per station |
| Format | CSV (`PRSA_Data_<Station>_20130301-20170228.csv`) |

### Variables

| Variable | Type | Description |
|---|---|---|
| PM2.5 | Pollutant | Fine particulate matter (μg/m³) |
| PM10 | Pollutant | Coarse particulate matter (μg/m³) |
| SO2 | Pollutant | Sulphur dioxide (μg/m³) |
| NO2 | Pollutant | Nitrogen dioxide (μg/m³) |
| CO | Pollutant | Carbon monoxide (μg/m³) |
| O3 | Pollutant | Ozone (μg/m³) |
| TEMP | Meteorological | Temperature (°C) |
| PRES | Meteorological | Atmospheric pressure (hPa) |
| DEWP | Meteorological | Dew point temperature (°C) |
| RAIN | Meteorological | Precipitation (mm) |
| WSPM | Meteorological | Wind speed (m/s) |
| wd | Meteorological | Wind direction (categorical) |

---

## Project Structure

```
CMP7005_PRAC1_Beijing_AQA/
│
├── CMP7005_PRAC1_Beijing_AQA.ipynb   # Main Colab notebook (all sections)
├── README.md                          # This file
│
└── data/                              # Place your downloaded CSV files here
    ├── PRSA_Data_Aotizhongxin_20130301-20170228.csv
    ├── PRSA_Data_Changping_20130301-20170228.csv
    ├── PRSA_Data_Dingling_20130301-20170228.csv
    ├── PRSA_Data_Dongsi_20130301-20170228.csv
    └── ...                            # (up to 12 station files)
```

> **Note:** The notebook writes `app.py` to `/content/app.py` at runtime — no separate app file needs to be committed.

---

## Getting Started

### Option 1 — Google Colab (recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run all**
3. In Section 2, upload your `PRSA_Data_*.csv` files when prompted
4. In Section 3, click the printed URL to open the Streamlit dashboard

### Option 2 — Local (Jupyter)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CMP7005_PRAC1_Beijing_AQA.git
cd CMP7005_PRAC1_Beijing_AQA

# Install dependencies
pip install pandas numpy scikit-learn seaborn plotly streamlit matplotlib

# Launch Jupyter
jupyter notebook CMP7005_PRAC1_Beijing_AQA.ipynb
```

---

## Notebook Sections

| Section | Description |
|---|---|
| **1 · Setup** | Install Streamlit, write `app.py` to disk with full source |
| **2 · Upload** | File picker for local CSV upload or Google Drive mount |
| **3 · App** | Launch interactive Streamlit dashboard via background thread |
| **4.1** | Library imports and consistent dark matplotlib theme |
| **4.2** | Station CSV loading with regex name extraction — Task 1 |
| **4.3** | Data understanding: shape, dtypes, missing value audit |
| **4.4** | Preprocessing pipeline: datetime, dedup, imputation, feature engineering |
| **4.5** | Statistical analysis and visualisations (EDA) — Task 2 |
| **4.6** | Feature preparation and train/test split for modelling |
| **4.7** | Model training, evaluation, and comparison — Task 3 |
| **4.8** | Summary printout with key findings — Task 4 |

---

## Streamlit App Pages

### 🏠 Home
Project overview, dataset metrics (12 stations, 35k records, 11 variables, 4 years), and variable reference tables.

### 📋 Dataset
Raw data preview, column information, statistical summary, missing-value bar chart, station distribution, and cleaned CSV download.

### 📍 EDA
Three-tab exploratory analysis:
- **Univariate** — histogram, box plot, violin, KDE+rug, ECDF
- **Bivariate** — scatter, OLS trend, hexbin density, joint KDE, group bar with Pearson r
- **Multivariate** — correlation heatmap, scatter matrix, parallel coordinates, 3D scatter, bubble chart, radar by station

### 📊 Visualisations
10 interactive chart types selectable from a dropdown:

| Chart | Description |
|---|---|
| Time Series | Multi-station, multi-variable line chart with hover |
| Monthly Heatmap | Year × month pivot heatmap |
| Wind Rose | Polar bar chart of average wind speed by direction |
| Pollutant Box Plots | Side-by-side box plots for all six pollutants |
| PM2.5 by Season & Station | Grouped bar chart |
| Rolling Average | Configurable moving average (6–720 hours) |
| Station Comparison Bar | Mean/median/max/min per station |
| AQI Category Distribution | Donut chart using US EPA PM2.5 breakpoints |
| Hourly Pattern | Mean value by hour, split by season |
| Correlation Network | Force-layout graph with configurable threshold |

### 🧪 Modelling
Interactive ML pipeline — select task type, features, target, scaler, test split, and CV folds. Run multiple models simultaneously. See results table, comparison chart, actual-vs-predicted scatter, residuals histogram, confusion matrix, ROC curve, and feature importance.

### 📝 Report
Auto-generated summary with dataset statistics, PM2.5 key metrics (mean, max, WHO exceedances), correlation bar chart, and academic references.

---

## Machine Learning Models

### Regression (target: PM2.5 μg/m³)

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularisation |
| Lasso Regression | L1 regularisation |
| Decision Tree | max_depth=8 |
| Random Forest | 50 estimators, n_jobs=-1 |
| Gradient Boosting | 50 estimators |
| SVR | Warning: slow on large datasets |

### Classification (target: high/low PM2.5 binarised at median)

| Model | Notes |
|---|---|
| Logistic Regression | max_iter=500 |
| Decision Tree | max_depth=8 |
| Random Forest | 50 estimators, n_jobs=-1 |
| Gradient Boosting | 50 estimators |
| K-Nearest Neighbours | k=5, n_jobs=-1 |
| Naive Bayes | GaussianNB |
| SVM | Warning: slow on large datasets |

### Preprocessing pipeline

1. Encode `season` and `station` with `LabelEncoder`
2. Scale features with `StandardScaler` (configurable)
3. 80/20 train-test split (`random_state=42`)
4. k-fold cross-validation (configurable, default 5)
5. Data capped at 20,000 rows for Colab runtime (configurable)

---

## Key Findings

1. **Winter dominates PM2.5 peaks** — coal-fired heating combined with thermal inversions causes the highest concentrations (December–February).
2. **Urban stations exceed suburban** — Dongsi and Guanyuan consistently show higher pollutant levels than the suburban Dingling and Huairou stations.
3. **Temperature negatively correlates with PM2.5** (Pearson r ≈ −0.30) — lower winter temperatures coincide with elevated particulate matter.
4. **NO2 and CO are the strongest predictors** — both are combustion by-products, confirming traffic and heating as primary sources.
5. **Dual daily peaks** — PM2.5 rises during morning rush hour (7–9 AM) and again in the evening (9–11 PM), reflecting traffic patterns.

---

## Dependencies

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
plotly>=5.13
streamlit>=1.20
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
```

> In Google Colab, only `streamlit` needs installing — all other packages are pre-installed.

---

## References

- Brauer, M. et al. (2021). Ambient particulate matter air pollution exposure and mortality. *Environmental Health Perspectives.*
- Li, Z. et al. (2019). Air pollution and health in China. *Environmental Research Letters.*
- Li, Z. et al. (2024). Recent trends in Beijing air quality. *Atmospheric Environment.*
- Lim, S.S. et al. (2020). Air quality and health burden. *The Lancet.*
- Sokhi, R.S. et al. (2022). Global air quality challenges. *npj Climate and Atmospheric Science.*
- Xu, J. & Zhang, Y. (2020). Emission controls in Beijing. *Science of the Total Environment.*

---

## Licence

This project was created for academic assessment purposes as part of **CMP7005 — Programming for Data Analysis** at Cardiff Metropolitan University. Not intended for commercial use.

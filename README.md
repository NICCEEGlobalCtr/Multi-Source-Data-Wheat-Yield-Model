# Code and Data for

## Coordinated multi-nutrient management enables sustainable wheat production

This repository contains the code used in the study **Coordinated multi-nutrient management enables sustainable wheat production**, submitted to ***[Nature Communications](https://www.nature.com/ncomms/)***.

---

## 1. Scripts

| Script | What it does |
|---|---|
| `SE_YP_model_main.py` | Trains and validates the yield prediction model (a stacking ensemble of XGBoost, LightGBM, Random Forest, and Linear Regression) and saves the final model. |
| `fig2_shap_pdp.py` | Calculates the SHAP importance and the partial dependence of the model. |
| `FP_survey_baseline_generate.py` | Builds the farmers' practice (FP) baseline of each site from the national farm survey. |
| `SE_YP_optimize.py` | Picks the (N, P₂O₅, K₂O) combination of each site under three strategies. **Agronomic optimal fertilization (AOF)** **maximizes predicted yield**. **Economic optimal fertilization (EOF)** **maximizes net farmer benefit (NFB)**, defined as grain revenue minus fertilizer cost. **Environmental-social optimal fertilization (ESOF)** **maximizes net ecological-economic benefit (NEEB)**, defined as NFB minus the environmental and social costs of N and P losses. |
| `SE_YP_bootstrap.py` | Quantifies the uncertainty of the recommendations and gains with 50 bootstrap replicates. |
| `FP_validation_and_county_totals.py` | Compares the predicted FP yields with the surveyed farm yields and calculates the collinearity among covariates and the area-weighted totals. |

---

## 2. Provided files

- **`test_data.csv`** is a small example dataset (20 plots) in the format that the model receives. It contains the site number `site`, the target column `yield`, and the 23 input variables of the model.
- **`SE_YP_model.pkl`** is the pre-trained stacking ensemble model. With this file you can predict yield directly without having to rerun `SE_YP_model_main.py` first.

---

## 3. Requirements

This code requires Python 3.12 and the following packages.

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm optuna joblib matplotlib shap openpyxl
```

---

## 4. How to run

### Quick start (using the pre-trained model)

Predict the yield of the plots in `test_data.csv`.

```python
import joblib
import numpy as np
import pandas as pd

data = pd.read_csv('test_data.csv')
model = joblib.load('SE_YP_model.pkl')
pred = np.expm1(model.predict(data.drop(columns=['site', 'yield'])))   # predicted yield (kg/ha)
```

### Full pipeline

The scripts below reproduce the full analysis and need the full dataset, which is not included in this repository. To use your own data, update the paths at the top of each script and run the scripts from the repository root in this order.

```bash
python SE_YP_model_main.py                  # train and validate the model
python fig2_shap_pdp.py                     # SHAP importance and partial dependence
python FP_survey_baseline_generate.py       # FP baseline
python SE_YP_optimize.py                    # AOF, EOF, and ESOF
python SE_YP_bootstrap.py                   # uncertainty of the recommendations
python FP_validation_and_county_totals.py   # FP validation and totals
```

`SE_YP_model_main.py` trains the stacking ensemble with the XGBoost hyperparameters fixed at the values selected by 50 trials of Optuna Bayesian search (5-fold CV grouped by site). The search itself is not rerun. The script validates the model with whole sites, spatial blocks, provinces, regions, and years held out, and saves the models to `results/SE-YP/models/`. Set `USE_GPU = False` at the top of `SE_YP_model_main.py` and `SE_YP_bootstrap.py` to run without a GPU.

`SE_YP_optimize.py` generates an (N, P₂O₅, K₂O) grid at 5 kg ha⁻¹ resolution from zero to the highest tested rate of each site, scores all combinations with the stacking model, and writes the optimal combination of each strategy per site to `results/SE-YP-opt/`.

---

## 5. Input data format

`test_data.csv` contains `site`, `yield`, and the 23 input variables of the model. These are the 12 site features and the 3 applied fertilizer rates listed below, the crop type dummies (`crop_type_sw`, `crop_type_ww`), the medium rates of the site (`N_med`, `P2O5_med`, `K20_med`), and the applied rates relative to the medium rates (`N_rel`, `P2O5_rel`, `K20_rel`). `SE_YP_model_main.py` calculates the last eight variables from the raw input.

The raw input files of the scripts must contain the following columns.

- **Target (training only)** `yield`
- **Site features** `Altitude`, `SOM`, `TN`, `AN`, `AP`, `AK`, `pH`, `irrigation`, `precipitation`, `Frost_free_day`, `temperature`, `Nsurplus`
- **Applied fertilizer (training only)** `N`, `P2O5`, `K20` (`K20` is just the column name for K₂O). Each site must contain all treatments of the experimental design, because the medium rate of each nutrient is taken from them.
- **Crop** `crop_type`, a categorical variable that will be one-hot encoded
- **IDs** `ID`, `No.`, `longitude`, `latitude`
- **For the optimization only** highest tested rates `Nmax`, `Pmax`, `Kmax` of each site (the search starts from zero), together with `Ndep` and `Region` (one of `North`, `Center`, `Yangtze River Plain`, `Southwest`). `Region` selects the N loss functions used by ESOF.

---

## 6. Fertilizer Prices

`EOF` and `ESOF` use default unit prices for grain, N, P₂O₅, and K₂O. Edit the constants at the top of `SE_YP_optimize.py` and `SE_YP_bootstrap.py` to use your own prices.

The N environmental damage function is region-specific (a sum of two exponentials plus a linear term plus an offset, parameterized separately for North, Center, Yangtze River Plain, and Southwest).

---

## 7. Computing Environment

- **Hardware**
  - NVIDIA RTX 5080 GPU

- **Key Software Versions**
  - Python 3.12
  - CUDA 12.8
  - PyTorch 2.7.1
  - XGBoost 2.1.4
  - LightGBM 4.5.0
  - scikit-learn 1.6.1
  - Optuna 4.2.1
  - SHAP 0.46.0
  - pandas 2.2.2
  - NumPy 2.0.2
  - matplotlib 3.9.2
  - seaborn 0.13.2
  - ArcGIS Pro 3.5.2 (Esri, Redlands, CA, USA)
  - Origin 2021 (OriginLab Corporation, Northampton, MA, USA)
  - SPSS 21 (IBM Corporation, Armonk, NY, USA)

---

## Citation

If you reference this work, please cite the associated manuscript.

> Keyu Ren<sup>&ast;</sup>, Hai Lan<sup>&ast;,#</sup>, Minggang Xu, Wenju Zhang, Eric A. Davidson, Xin Zhang<sup>#</sup>, Yinghua Duan<sup>#</sup>. Coordinated multi-nutrient management enables sustainable wheat production. *[Nature Communications](https://www.nature.com/ncomms/)* (submitted).
> 
> <sup>&ast;</sup>These authors contributed equally.  
> <sup>#</sup>Authors to whom any correspondence should be addressed.

Full citation details will be updated upon publication.

# Code and Data for

## Coordinated multi-nutrient management enables sustainable wheat production

This repository contains the code used in the study **Coordinated multi-nutrient management enables sustainable wheat production**, submitted to ***[Nature Communications](https://www.nature.com/ncomms/)***.

---

## 1. Scripts

| Script | What it does |
|---|---|
| `modeling.py` | Trains the yield prediction model (tuned XGBoost plus a stacking ensemble of XGBoost, LightGBM, Random Forest, and Linear Regression). |
| `AOF_optimization.py` | **Agronomic Optimization of Fertilization (AOF)** picks the (N, P, K) combination that **maximizes predicted yield** for each site. |
| `EOF_optimization.py` | **Economic Optimization of Fertilization (EOF)** picks the combination that **maximizes farm revenue** (yield value minus fertilizer cost). |
| `ESOF_optimization.py` | **Economic and Social Optimization of Fertilization (ESOF)** picks the combination that **maximizes Net Ecosystem Economic Benefit (NEEB)**, defined as revenue minus fertilizer cost minus monetized environmental damage from N and P losses. |

---

## 2. Provided files

- **`test_training_data.csv`** is a small example dataset for testing `modeling.py`. It contains the target column `yield` together with all site features, applied fertilizer rates, and crop type.
- **`test_optimization_data.csv`** is a small example dataset for testing the three optimization scripts. In addition to the site features and crop type, it includes the per-site fertilizer search bounds (`Nmin`/`Nmax`, `Pmin`/`Pmax`, `Kmin`/`Kmax`) along with `Ndep` and `Region`, which are needed by ESOF.
- **`model_stacking.pkl`** is the pre-trained stacking ensemble model used by all three optimization scripts. With this file you can run the optimizations directly without having to rerun `modeling.py` first.

---

## 3. Requirements

This code requires Python 3.12 and the following packages.

```bash
pip install numpy pandas scikit-learn xgboost lightgbm optuna joblib matplotlib seaborn shap geopandas shapely
```

---

## 4. How to run

### Quick start (using the pre-trained model)

Run any of the optimization scripts on `test_optimization_data.csv`.

```bash
python AOF_optimization.py    # maximize yield
python EOF_optimization.py    # maximize revenue (top 10 combos per site)
python ESOF_optimization.py   # maximize NEEB     (top 10 combos per site)
```

Each script reads the input CSV, generates an (N, P, K) grid at 5 kg ha⁻¹ resolution within each site's bounds, scores all combinations with the stacking model, and writes the best combinations per site to `predictions/`. To use your own data, update the `data_path` variable at the top of each script.

### Retrain the model (optional)

To retrain the stacking model, run `modeling.py` with `test_training_data.csv` (or your own training data, after updating the `data_path` variable at the top of the script).

```bash
python modeling.py
```

This runs 50 trials of Optuna Bayesian hyperparameter search on XGBoost (5-fold CV), trains the stacking ensemble, and saves both models to `model/`.

---

## 5. Input data format

The input CSV must contain the following columns.

- **Target (training only)** `yield`
- **Site features** `Altitude`, `SOM`, `TN`, `AN`, `AP`, `AK`, `pH`, `irrigation`, `precipitation`, `Frost_free_day`, `temperature`, `Nsurplus`
- **Applied fertilizer (training only)** `N`, `P2O5`, `K20` (`K20` is just the column name for K₂O)
- **Crop** `crop_type`, a categorical variable that will be one-hot encoded
- **IDs** `ID`, `No.`, `longitude`, `latitude`
- **For optimization scripts only** per-site fertilizer search bounds `Nmin`, `Nmax`, `Pmin`, `Pmax`, `Kmin`, `Kmax`
- **For ESOF only** `Ndep`, `Region` (one of `North`, `Center`, `Yangtze River Plain`, `Southwest`)

---

## 6. Fertilizer Prices

`EOF` and `ESOF` use default unit prices for grain, N, P₂O₅, and K₂O. Edit the constants at the top of `EOF_optimization.py` and `ESOF_optimization.py` to use your own prices.

The N environmental damage function in `ESOF_optimization.py` is region-specific (a sum of two exponentials plus a linear term plus an offset, parameterized separately for North, Center, Yangtze River Plain, and Southwest).

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
  - scikit-learn 1.5.2
  - Optuna 3.6.1
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

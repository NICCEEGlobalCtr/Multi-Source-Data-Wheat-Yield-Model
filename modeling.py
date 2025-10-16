import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import joblib

file_path = 'data/data.csv'
data = pd.read_csv(file_path)

if 'crop_type' in data.columns:
    data = pd.get_dummies(data, columns=['crop_type'])

id_info = data[['ID', 'No.', 'longitude', 'latitude']]
excluded_columns = ['longitude', 'latitude']
data_numeric = data.drop(columns=excluded_columns, errors='ignore')

data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = data_numeric.select_dtypes(include=['number']).columns
data_numeric[numeric_cols] = data_numeric[numeric_cols].fillna(data_numeric[numeric_cols].mean())
non_numeric_cols = data_numeric.select_dtypes(exclude=['number']).columns
for col in non_numeric_cols:
    data_numeric[col] = data_numeric[col].fillna(data_numeric[col].mode()[0])

base_features = [
    'Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH',
    'irrigation', 'precipitation', 'Frost_free_day',
    'temperature', 'N', 'P2O5', 'K20', 'Nsurplus'
]
crop_type_columns = [col for col in data_numeric.columns if col.startswith('crop_type_')]
feature_columns = base_features + crop_type_columns

X = data_numeric[feature_columns]
Y = np.log1p(data_numeric['yield'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'hist'
    }
    model = xgb.XGBRegressor(**param)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, Y_train, cv=cv,
                             scoring='neg_mean_squared_error', n_jobs=-1)
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best parameters from Bayesian optimization:", study.best_params)
print("Best CV score (negative MSE):", study.best_value)

best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0, tree_method='hist')
best_xgb.fit(X_train, Y_train)

joblib.dump(best_xgb, "model/model_best_xgb.pkl")
print("XGBoost model saved.")

Y_pred_xgb_train = best_xgb.predict(X_train)
Y_train_orig = np.expm1(Y_train)
Y_pred_xgb_train_orig = np.expm1(Y_pred_xgb_train)
mse_xgb_train = mean_squared_error(Y_train_orig, Y_pred_xgb_train_orig)
rmse_xgb_train = np.sqrt(mse_xgb_train)
mae_xgb_train = mean_absolute_error(Y_train_orig, Y_pred_xgb_train_orig)
r2_xgb_train = r2_score(Y_train_orig, Y_pred_xgb_train_orig)
print("XGBoost training metrics:")
print("MSE:", mse_xgb_train, "RMSE:", rmse_xgb_train, "MAE:", mae_xgb_train, "R2:", r2_xgb_train)

Y_pred_xgb_test = best_xgb.predict(X_test)
Y_test_orig = np.expm1(Y_test)
Y_pred_xgb_test_orig = np.expm1(Y_pred_xgb_test)
mse_xgb_test = mean_squared_error(Y_test_orig, Y_pred_xgb_test_orig)
rmse_xgb_test = np.sqrt(mse_xgb_test)
mae_xgb_test = mean_absolute_error(Y_test_orig, Y_pred_xgb_test_orig)
r2_xgb_test = r2_score(Y_test_orig, Y_pred_xgb_test_orig)
print("\nXGBoost test metrics:")
print("MSE:", mse_xgb_test, "RMSE:", rmse_xgb_test, "MAE:", mae_xgb_test, "R2:", r2_xgb_test)

importance = best_xgb.feature_importances_
plt.figure(figsize=(10, 8))
plt.barh(X_train.columns, importance)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

estimators = [
    ('xgb', best_xgb),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    ('lr', LinearRegression())
]
stacking_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(),
                                 cv=5, n_jobs=-1)
stacking_reg.fit(X_train, Y_train)

joblib.dump(stacking_reg, "model/model_stacking.pkl")
print("Stacking Ensemble model saved.")

Y_pred_stack_train = stacking_reg.predict(X_train)
Y_pred_stack_train_orig = np.expm1(Y_pred_stack_train)
mse_stack_train = mean_squared_error(Y_train_orig, Y_pred_stack_train_orig)
rmse_stack_train = np.sqrt(mse_stack_train)
mae_stack_train = mean_absolute_error(Y_train_orig, Y_pred_stack_train_orig)
r2_stack_train = r2_score(Y_train_orig, Y_pred_stack_train_orig)
print("\nStacking Ensemble training metrics:")
print("MSE:", mse_stack_train, "RMSE:", rmse_stack_train, "MAE:", mae_stack_train, "R2:", r2_stack_train)

Y_pred_stack_test = stacking_reg.predict(X_test)
Y_pred_stack_test_orig = np.expm1(Y_pred_stack_test)
mse_stack_test = mean_squared_error(Y_test_orig, Y_pred_stack_test_orig)
rmse_stack_test = np.sqrt(mse_stack_test)
mae_stack_test = mean_absolute_error(Y_test_orig, Y_pred_stack_test_orig)
r2_stack_test = r2_score(Y_test_orig, Y_pred_stack_test_orig)
print("\nStacking Ensemble test metrics:")
print("MSE:", mse_stack_test, "RMSE:", rmse_stack_test, "MAE:", mae_stack_test, "R2:", r2_stack_test)

results = pd.DataFrame({
    'ID': data.loc[X_test.index, 'ID'],
    'No.': data.loc[X_test.index, 'No.'],
    'longitude': data.loc[X_test.index, 'longitude'],
    'latitude': data.loc[X_test.index, 'latitude'],
    'actual_yield': Y_test_orig,
    'pred_yield': Y_pred_xgb_test_orig
})
results['absolute_error'] = np.abs(results['actual_yield'] - results['pred_yield'])
results['relative_error'] = results['absolute_error'] / results['actual_yield']
results['squared_error'] = (results['actual_yield'] - results['pred_yield']) ** 2
results['percentage_error'] = results['relative_error'] * 100
mean_abs_error = results['absolute_error'].mean()
std_abs_error = results['absolute_error'].std()
results['normalized_error'] = (results['absolute_error'] - mean_abs_error) / std_abs_error
epsilon = 1e-8
results['log_error'] = np.abs(np.log(results['pred_yield'] + epsilon) - np.log(results['actual_yield'] + epsilon))
results['residual'] = results['pred_yield'] - results['actual_yield']
print(results.head())
results.to_csv("validation/validation_metrics.csv", index=False)

geometry = [Point(xy) for xy in zip(results['longitude'], results['latitude'])]
gdf = gpd.GeoDataFrame(results, geometry=geometry, crs="EPSG:4326")
gdf.to_file("validation/shp/validation_metrics_map.shp")

mbe = results['residual'].mean()
median_res = results['residual'].median()
plt.figure(figsize=(10, 6))
sns.histplot(results['residual'], bins=30, kde=True)
plt.axvline(mbe, color='red', linestyle='--', label=f'MBE = {mbe:.2f}')
plt.axvline(median_res, color='blue', linestyle='--', label=f'Median = {median_res:.2f}')
plt.xlabel("Residual (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.legend()
plt.show()

# 🚗 Car Price Prediction Machine Learning Pipeline

An end-to-end Machine Learning project to predict used car prices based on vehicle specifications, categorical features, and historical sales data.

---

## 📌 Dataset

* **Source**: [Kaggle - Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge/data)
* **Dataset Size**: 19,237 rows x 18 columns

---

## 🛠️ Machine Learning Workflow

### 1. Feature Preprocessing & Engineering
* **Categorical Encoding:**
  * **Low-cardinality features** ($\le 20$ unique values): Encoded using `OneHotEncoder(handle_unknown='ignore', drop='first')`.
  * **High-cardinality features** ($> 20$ unique values): Encoded using 5-fold cross-validated `TargetEncoder(smooth='auto')`.
* **Numerical Scaling:** Standardized via `StandardScaler`.
* **Target Transformation:** Applied log transformation $\log(1 + x)$ via `np.log1p()` to handle target skewness and prevent negative predictions. Back-transformed during evaluation using `np.expm1()`.

### 2. Baseline Models Performance (Holdout Test Set)

| Model | Target Space Metrics | Log Space Metrics |
| :--- | :--- | :--- |
| **Linear Regression** | $R^2: 0.4339$, $\text{RMSE}: 13,870.68$ | $R^2: 0.4468$, $\text{RMSE}: 0.6633$ |
| **Ridge Regression** | $R^2: 0.4338$, $\text{RMSE}: 13,871.16$ | $R^2: 0.4467$, $\text{RMSE}: 0.6633$ |
| **Random Forest Regressor** | **$R^2: 0.6396$, $\text{RMSE}: 11,067.01$** | **$R^2: 0.7510$, $\text{RMSE}: 0.4450$** |

### 3. Model Comparison (5-Fold Cross-Validation)

To prevent data leakage during preprocessing, `scikit-learn` `Pipeline` objects were evaluated using 5-Fold Cross-Validation:

| Model / Configuration | Mean $R^2$ | Std $R^2$ | Mean RMSE | Std RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.4456 | 0.0098 | 0.6502 | 0.0139 |
| **Ridge Regression** | 0.4456 | 0.0097 | 0.6502 | 0.0138 |
| **Random Forest (Default)** | 0.7571 | 0.0044 | 0.4304 | 0.0093 |
| **Random Forest (`max_depth=10`, `min_samples_split=10`)** | 0.6561 | 0.0094 | 0.5121 | 0.0115 |
| **Random Forest (`max_depth=100`, `min_samples_split=10`)** | 0.7478 | 0.0069 | 0.4385 | 0.0092 |
| **Random Forest (Tuned Final Model)** | **0.7680** | **0.0077** | **0.4207** | **0.0132** |

### 4. Hyperparameter Tuning

Executed `RandomizedSearch` over a 5-fold cross-validation pipeline across the hyperparameter space:

* **Search Space:**
  * `n_estimators`: `[100, 200, 300]`
  * `max_depth`: `[None, 20, 30]`
  * `min_samples_split`: `[2, 5, 10]`
  * `min_samples_leaf`: `[1, 2, 4]`
  * `max_features`: `['sqrt', 'log2', None]`

* **Best Hyperparameters Found:**
  ```python
  {
      'n_estimators': 300,
      'max_depth': 30,
      'min_samples_split': 2,
      'min_samples_leaf': 1,
      'max_features': 'sqrt'
  }

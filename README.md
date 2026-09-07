# Daily Birth Forecasting - LSTM vs ARIMA

**Short project summary**

This repository compares a simple single-layer LSTM and an ARIMA model to forecast *daily female births in California (1959)*. The goal is to demonstrate practical time-series modeling approaches for public-health planning and to provide reproducible code, evaluation, and notes on model optimization.

---

## 🔎 Dataset

* **File:** `daily-total-female-births-CA.csv`
* **Description:** Daily total female births in California for the year 1959 (365 rows).
* **Columns:**

  * `date` — calendar date (YYYY-MM-DD)
  * `births` — number of female births on that date

Summary statistics (dataset used in this repo):

* Rows: 365
* Mean births: ~41.98
* No missing values

---



## Quick results

**Evaluation on test set (80/20 time-based split)**

| Model                 |      MAE |     RMSE |       MAPE |
| --------------------- | -------: | -------: | ---------: |
| LSTM (single-layer)   | **5.38** | **6.66** | **12.54%** |
| ARIMA (p=5, d=1, q=0) |     5.58 |     6.84 |     14.08% |

> The LSTM performed slightly better than ARIMA across MAE, RMSE and MAPE in this experiment.

---

## What’s in the code (high-level)

1. **Data loading & exploration** (`data_load.py` / `notebooks/exploration.ipynb`)

   * Load CSV into a pandas DataFrame.
   * Convert `date` to `datetime` and inspect basic statistics and distribution of `births`.

2. **Data preparation** (`prepare.py`)

   * Create `date_ordinal` via `date.toordinal()` for a numeric representation of dates.
   * Build sliding windows for LSTM with `timesteps=5`.
   * Perform an 80/20 chronological train/test split (no shuffling for time-series).

3. **LSTM modeling** (`train_lstm.py`)

   * A single-layer LSTM with `units=50`, `activation='relu'`, `optimizer=Adam(lr=0.001)`, trained for 30–50 epochs with `batch_size=32`.
   * Input shape: `(timesteps=5, features=1)`.
   * Model saved as `models/optimized_lstm_birth_model.keras`.

4. **ARIMA modeling** (`train_arima.py`)

   * An ARIMA(p=5, d=1, q=0) is fit on the training target series and saved as `models/arima_birth_model.pkl`.

5. **Evaluation & visualization** (`evaluate.py`, `visualize.py`)

   * Metrics: MAE, RMSE, MAPE.
   * Actual vs predicted plots for both train and test sets.

---



## Notes on hyperparameter tuning

* An attempt to use **GridSearchCV** (via `scikeras.wrappers.KerasRegressor`) led to repeated `AttributeError` / compatibility and shape issues.AttributeError or shape errors during cross-validation (since LSTM input must be 3D, but GridSearchCV re-splits it without maintaining shape). Very slow training for each grid combination — because LSTMs are expensive to train repeatedly. Instability in results due to random weight initialization.

* **Fallback approach used:** manual/human-guided tuning — controlled experiments varying `units`, `learning_rate`, `batch_size`, `epochs`, and `activation`, monitoring validation MAE/RMSE/ MAP E for stability.

* **Recommended automated alternatives** (future work): use Keras Tuner, Optuna, or Ray Tune which integrate more cleanly with Keras/TensorFlow, support early-stopping, and consume fewer resources via smarter search strategies (bayesian/Hyperband).

---

## Next steps & ideas

* Add **feature engineering**: day-of-week, month, holidays, lag features, rolling statistics.
* Try **seasonal models** (SARIMA) since yearly seasonality may exist.
* Use advanced models: **Transformer**, **Temporal Convolutional Networks**, or ensemble multiple models.
* Apply **cross-validation for time-series** (TimeSeriesSplit) for robust evaluation.
* Improve hyperparameter tuning with **Keras Tuner** or **Optuna**.

---


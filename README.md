# Daily Birth Forecasting — LSTM vs ARIMA

A time-series forecasting project comparing a **single-layer LSTM** with an **ARIMA** model for predicting daily female births in California during 1959.

The project focuses on building a simple and explainable forecasting pipeline, evaluating both models using standard regression metrics, and analyzing their predictions on unseen future data.

---

## Project Overview

The objective is to forecast the number of female births for a given day using historical birth records.

Two different approaches are compared:

* **LSTM** — Deep learning model designed to learn patterns from sequential data.
* **ARIMA** — Classical statistical model commonly used for time-series forecasting.

The models are evaluated on the same time-based test set to compare their forecasting performance.

---

## Dataset

**Dataset:** Daily Female Births in California, 1959

* **Rows:** 365
* **Features:** Date, Births
* **Target:** Daily number of female births
* **Missing Values:** None
* **Average Daily Births:** ~41.98

The dataset contains one year of daily female birth counts recorded in California during 1959.

---

## Project Pipeline

```text
Daily Birth Data
        ↓
Data Loading & Exploration
        ↓
Data Cleaning
        ↓
Time-Series Visualization
        ↓
Chronological Train/Test Split
        ↓
Sliding Window Creation
        ↓
        ┌───────────────┐
        ↓               ↓
      LSTM            ARIMA
        ↓               ↓
        └───────┬───────┘
                ↓
       Model Evaluation
                ↓
       MAE / RMSE / MAPE
                ↓
     Actual vs Predicted
                ↓
      Model Comparison
```

---

## Data Preparation

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted the `date` column into datetime format.
3. Checked the dataset for missing values and basic statistics.
4. Used an **80/20 chronological train-test split** to prevent future information from being used during training.
5. Created **5-day sliding windows** for the LSTM model.
6. Scaled the time-series values before training the LSTM.

---

## LSTM Model

A simple single-layer LSTM was implemented for forecasting.

### Architecture

```text
Input Sequence (5 days)
        ↓
LSTM Layer (50 units)
        ↓
Dense Output Layer
        ↓
Predicted Birth Count
```

### Configuration

* LSTM layers: 1
* LSTM units: 50
* Timesteps: 5
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 32
* Epochs: 30–50
* Activation: ReLU

The LSTM uses the birth counts from previous days to predict the birth count for the next day.

---

## ARIMA Model

ARIMA was implemented as a classical time-series baseline.

**Configuration:**

* Model: ARIMA(5,1,0)
* `p = 5` → autoregressive terms
* `d = 1` → first-order differencing
* `q = 0` → moving-average terms

The ARIMA model was trained on the same training portion of the time series and evaluated on the future test observations.

---

## Results

Evaluation was performed on the **20% time-based test set**.

| Model        |      MAE |     RMSE |       MAPE |
| ------------ | -------: | -------: | ---------: |
| **LSTM**     | **5.38** | **6.66** | **12.54%** |
| ARIMA(5,1,0) |     5.58 |     6.84 |     14.08% |

### Observation

The LSTM achieved slightly better performance than ARIMA across all three evaluation metrics in this experiment.

* Lower **MAE**
* Lower **RMSE**
* Lower **MAPE**

This indicates that the LSTM produced slightly more accurate forecasts on the held-out test period.

---

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Lower MAE → Better predictions**

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily than MAE.

**Lower RMSE → Better predictions**

### MAPE — Mean Absolute Percentage Error

Measures prediction error as a percentage of the actual value.

**Lower MAPE → Better forecasting performance**

---

## Visualization

The project includes visualizations for:

* Daily birth trends
* Training and validation loss
* Actual vs predicted births
* LSTM vs ARIMA predictions
* Test-set forecasting performance

---

## Hyperparameter Experiments

Different LSTM configurations were experimented with by varying:

* Number of LSTM units
* Learning rate
* Batch size
* Number of epochs
* Activation function

The final configuration was selected based on validation performance and training stability.

Automated `GridSearchCV` was not used for the final model because LSTM inputs require a 3D sequence structure, making standard cross-validation less convenient. Manual experimentation was therefore used for model selection.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Statsmodels
* Jupyter Notebook

---

## Project Structure

```text
Daily-Birth-Forecasting/
│
├── data/
│   └── daily-total-female-births-CA.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── data_load.py
├── prepare.py
├── train_lstm.py
├── train_arima.py
├── evaluate.py
├── visualize.py
│
├── models/
│   ├── optimized_lstm_birth_model.keras
│   └── arima_birth_model.pkl
│
└── README.md
```

---

## Key Takeaways

* Time-series data requires **chronological splitting** rather than random shuffling.
* LSTM can learn temporal dependencies from sequential observations.
* ARIMA provides a strong and interpretable classical forecasting baseline.
* MAE, RMSE, and MAPE provide complementary views of forecasting accuracy.
* In this experiment, **LSTM slightly outperformed ARIMA** on the held-out test set.

---

## Future Improvements

* Add lag-based and rolling statistical features.
* Investigate seasonal forecasting using SARIMA.
* Experiment with different sequence lengths.
* Compare deeper LSTM architectures.
* Apply proper time-series cross-validation.
* Explore GRU and Transformer-based forecasting models.

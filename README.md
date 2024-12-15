# BatteryLifetimePredictor Documentation

## Overview
The `BatteryLifetimePredictor` is a machine learning-based system designed to:
- Detect faults in battery cells.
- Predict the remaining lifetime of battery cells.
- Recommend backup cells if needed.

This system processes raw sensor data to generate meaningful features and uses them to train three models: a fault detector, a lifetime predictor, and a backup requirement predictor.

---

## Key Components

### Attributes
- **`fault_detector`**: A `RandomForestClassifier` used for fault detection.
- **`life_predictor`**: A `RandomForestRegressor` used for predicting the remaining life of a battery.
- **`backup_predictor`**: A `RandomForestRegressor` used to estimate the number of backup cells required.
- **`scaler`**: A `StandardScaler` for normalizing feature data.
- **`history`**: A dictionary to store historical training or prediction logs.

### Methods

#### 1. `__init__()`
Initializes the class and its models.

#### 2. `process_raw_data(df)`
**Purpose**: Processes raw CSV data to generate cell-wise time-series features.

**Parameters:**
- `df`: A Pandas DataFrame containing raw data with columns:
  - `timestamp` (datetime)
  - `group_id` (int)
  - `cell_id` (int)
  - `cell_voltage` (float)
  - `cell_temperature` (float)
  - `cell_status` (int)

**Returns:**
- A processed DataFrame containing calculated features:
  - `voltage_mean`
  - `voltage_std`
  - `temperature_mean`
  - `temperature_max`
  - `voltage_trend`
  - `temp_stress`
  - `voltage_stability`
  - `time_span`
  - `status`

#### 3. `extract_features(cell_data)`
**Purpose**: Extracts features for prediction from processed data.

**Parameters:**
- `cell_data`: A Pandas Series or dictionary containing preprocessed feature data for a single cell.

**Returns:**
- A NumPy array of feature values.

#### 4. `train(data_path)`
**Purpose**: Trains the models using processed data from a CSV file.

**Parameters:**
- `data_path`: Path to the CSV file containing raw data.

**Workflow:**
1. Reads and processes the data using `process_raw_data`.
2. Extracts features and labels.
3. Scales features using `StandardScaler`.
4. Trains three models:
   - `fault_detector` using fault labels.
   - `life_predictor` using remaining life labels.
   - `backup_predictor` using backup requirements.

#### 5. `predict(cell_data)`
**Purpose**: Predicts the health, remaining life, and backup needs for a cell.

**Parameters:**
- `cell_data`: A dictionary containing cell features.

**Returns:**
- A dictionary with the following predictions:
  - `fault_probability`: Probability of a fault.
  - `remaining_life_hours`: Predicted remaining life in hours.
  - `backup_cells_needed`: Recommended number of backup cells.
  - `status`: A status message indicating health.

#### 6. `_get_status(fault_prob, remaining_life)`
**Purpose**: Generates a status message based on fault probability and remaining life.

**Parameters:**
- `fault_prob`: Probability of a fault.
- `remaining_life`: Predicted remaining life in hours.

**Returns:**
- A string indicating cell health status.
  - `"Critical - Immediate Replacement Needed"` if `fault_prob > 0.7`.
  - `"Warning - Plan Replacement"` if `fault_prob > 0.3` or `remaining_life < 100`.
  - `"Healthy"` otherwise.

#### 7. `save_model(filename='battery_predictor.pkl')`
**Purpose**: Saves the trained models and scaler to a file.

**Parameters:**
- `filename`: Name of the file to save the models.

#### 8. `load_model(filename='battery_predictor.pkl')`
**Purpose**: Loads the models and scaler from a file.

**Parameters:**
- `filename`: Name of the file containing the models.

**Returns:**
- An instance of `BatteryLifetimePredictor` with loaded models.

---

## Example Usage

### Training the Models
```python
predictor = BatteryLifetimePredictor()
predictor.train('battery_data.csv')
```

### Making Predictions
```python
test_cell = {
    'voltage_mean': 3.8,
    'voltage_std': 0.1,
    'temperature_mean': 30,
    'temperature_max': 35,
    'voltage_trend': -0.001,
    'temp_stress': 0.1,
    'voltage_stability': 0.05,
    'time_span': 100
}
prediction = predictor.predict(test_cell)
print(prediction)
```

### Saving and Loading Models
```python
# Save the trained models
predictor.save_model()

# Load the models
loaded_predictor = BatteryLifetimePredictor.load_model()
```

---

## Requirements
- Python 3.7+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`

---

## Notes
- Ensure data preprocessing is consistent between training and prediction.
- Adjust thresholds in `_get_status` for specific use cases.
- Extend feature engineering in `process_raw_data` to improve model performance.

---

## Future Improvements
- Add support for additional models or metrics.
- Automate hyperparameter tuning.
- Implement real-time streaming support for predictions.


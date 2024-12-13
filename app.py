"""
Battery Fault Detection API

This module implements a Flask-based REST API for battery fault detection.
It provides endpoints for real-time fault prediction using a pre-trained
machine learning model.

The API accepts battery cell measurements (voltage, temperature, time)
and returns fault predictions with confidence scores.
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Load the trained model
model = joblib.load("./PKL/battery_fault_detector.pkl")

def extract_features(data):
    """Extract features from battery cell time series data.
    
    Args:
        data (pd.DataFrame): DataFrame containing battery measurements with columns:
            - voltage: Array of voltage measurements
            - temperature: Array of temperature measurements
            - time: Array of timestamps

    Returns:
        np.array: 8-dimensional feature vector containing:
            - Voltage statistics (mean, std, min, max, range)
            - Temperature mean
            - Voltage rate of change statistics (max, mean)
    """
    voltage = np.array(data['voltage'])
    temp = np.array(data['temperature'])
    time = np.array(data['time'])
    
    # Voltage features
    v_mean = np.mean(voltage)
    v_std = np.std(voltage)
    v_min = np.min(voltage)
    v_max = np.max(voltage)
    v_range = v_max - v_min
    
    # Temperature features 
    t_mean = np.mean(temp)
    
    # Rate of change
    voltage_diff = np.diff(voltage)
    max_roc = np.max(np.abs(voltage_diff))
    mean_roc = np.mean(np.abs(voltage_diff))
    
    return np.array([
        v_mean, v_std, v_min, v_max, v_range,
        t_mean, max_roc, mean_roc
    ])

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_fault():
    """Predict battery fault from cell measurements.
    
    Endpoint: POST /predict
    
    Request Body (JSON):
    {
        "time": [t1, t2, ...],        # Array of timestamps
        "voltage": [v1, v2, ...],     # Array of voltage measurements
        "temperature": [temp1, ...]    # Array of temperature measurements
    }
    
    Returns:
        JSON object with:
            - fault_detected: boolean indicating fault presence
            - confidence: prediction probability (0-1)
            - status: string description of prediction
    
    Error Responses:
        - 400: Missing required fields
        - 500: Internal processing error
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['time', 'voltage', 'temperature']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
                
        # Convert to DataFrame for feature extraction
        df = pd.DataFrame(data)
        
        # Extract features
        features = extract_features(df)
        
        # Make prediction
        prediction = model.predict(features.reshape(1, -1))[0]
        probability = model.predict_proba(features.reshape(1, -1))[0]
        
        return jsonify({
            'fault_detected': bool(prediction),
            'confidence': float(max(probability)),
            'status': 'Fault detected' if prediction == 1 else 'Healthy'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    """Run the Flask application.
    
    The API server runs on localhost:5000 by default.
    Debug mode is enabled for development purposes.
    """
    app.run(debug=True, port=5000)
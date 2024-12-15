import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

class BatteryLifetimePredictor:
    def __init__(self):
        self.fault_detector = RandomForestClassifier(n_estimators=100, random_state=42)
        self.life_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.backup_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.history = {}
        
    def process_raw_data(self, df):
        """Process raw CSV data into cell-wise time series."""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by cell and calculate features
        cell_data = []
        
        for (group_id, cell_id), group in df.groupby(['group_id', 'cell_id']):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate time-based features
            voltage_trend = np.polyfit(range(len(group)), group['cell_voltage'], 1)[0]
            temp_stress = np.sum(group['cell_temperature'] > 40) / len(group)
            voltage_stability = group['cell_voltage'].std()
            
            # Calculate cell age (in hours)
            time_span = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600
            
            cell_data.append({
                'group_id': group_id,
                'cell_id': cell_id,
                'voltage_mean': group['cell_voltage'].mean(),
                'voltage_std': group['cell_voltage'].std(),
                'temperature_mean': group['cell_temperature'].mean(),
                'temperature_max': group['cell_temperature'].max(),
                'voltage_trend': voltage_trend,
                'temp_stress': temp_stress,
                'voltage_stability': voltage_stability,
                'time_span': time_span,
                'status': group['cell_status'].mode()[0]
            })
            
        return pd.DataFrame(cell_data)

    def extract_features(self, cell_data):
        """Extract features for prediction."""
        features = [
            cell_data['voltage_mean'],
            cell_data['voltage_std'],
            cell_data['temperature_mean'],
            cell_data['temperature_max'],
            cell_data['voltage_trend'],
            cell_data['temp_stress'],
            cell_data['voltage_stability'],
            cell_data['time_span']
        ]
        return np.array(features)

    def train(self, data_path):
        """Train models using processed data."""
        # Read and process raw data
        raw_df = pd.read_csv(data_path)
        processed_df = self.process_raw_data(raw_df)
        
        # Prepare features
        X = []
        y_fault = []
        y_life = []
        y_backup = []
        
        for _, row in processed_df.iterrows():
            features = self.extract_features(row)
            X.append(features)
            
            # Define labels
            is_fault = row['status'] > 1  # Status > 1 indicates fault
            remaining_life = max(0, 1000 - row['time_span']) if row['status'] <= 1 else 0
            backup_needed = 1 if row['status'] > 1 else 0
            
            y_fault.append(is_fault)
            y_life.append(remaining_life)
            y_backup.append(backup_needed)
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.fault_detector.fit(X_scaled, y_fault)
        self.life_predictor.fit(X_scaled, y_life)
        self.backup_predictor.fit(X_scaled, y_backup)

    def predict(self, cell_data):
        """Predict cell health and backup requirements."""
        features = self.extract_features(cell_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        fault_prob = self.fault_detector.predict_proba(features_scaled)[0][1]
        remaining_life = self.life_predictor.predict(features_scaled)[0]
        backup_needed = self.backup_predictor.predict(features_scaled)[0]
        
        return {
            'fault_probability': float(fault_prob),
            'remaining_life_hours': float(remaining_life),
            'backup_cells_needed': int(np.ceil(backup_needed)),
            'status': self._get_status(fault_prob, remaining_life)
        }
    
    def _get_status(self, fault_prob, remaining_life):
        if fault_prob > 0.7:
            return 'Critical - Immediate Replacement Needed'
        elif fault_prob > 0.3 or remaining_life < 100:
            return 'Warning - Plan Replacement'
        return 'Healthy'

    def save_model(self, filename='battery_predictor.pkl'):
        """Save all models to single file."""
        model_data = {
            'fault_detector': self.fault_detector,
            'life_predictor': self.life_predictor,
            'backup_predictor': self.backup_predictor,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename='battery_predictor.pkl'):
        """Load model from file."""
        predictor = cls()
        model_data = joblib.load(filename)
        predictor.fault_detector = model_data['fault_detector']
        predictor.life_predictor = model_data['life_predictor']
        predictor.backup_predictor = model_data['backup_predictor']
        predictor.scaler = model_data['scaler']
        return predictor

if __name__ == "__main__":
    # Initialize and train model
    predictor = BatteryLifetimePredictor()
    print("Training models...")
    predictor.train('battery_data.csv')
    
    # Test prediction on sample data
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
    
    # Make prediction
    prediction = predictor.predict(test_cell)
    print("\nTest Prediction Results:")
    for key, value in prediction.items():
        print(f"{key}: {value}")
    
    # Save model
    predictor.save_model()
    print("\nModel saved successfully")
    
    # Verify model loading
    loaded_predictor = BatteryLifetimePredictor.load_model()
    loaded_prediction = loaded_predictor.predict(test_cell)
    print("\nVerification - Loaded Model Prediction:")
    for key, value in loaded_prediction.items():
        print(f"{key}: {value}")
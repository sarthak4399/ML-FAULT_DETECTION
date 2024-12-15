
import joblib
import numpy as np
import pandas as pd

def load_models():
    damage_model = joblib.load('models/damage_model.pkl')
    damage_scaler = joblib.load('models/damage_scaler.pkl')
    backup_model = joblib.load('models/backup_model.pkl')
    backup_scaler = joblib.load('models/backup_scaler.pkl')
    return damage_model, damage_scaler, backup_model, backup_scaler

def prepare_features(cell_data):
    # Create additional features
    cell_data['voltage_current_ratio'] = cell_data['voltage'] / (abs(cell_data['current']) + 1e-6)
    cell_data['power'] = cell_data['voltage'] * cell_data['current']
    
    features = ['voltage', 'current', 'temperature', 'soc', 
                'voltage_current_ratio', 'power']
    return cell_data[features]

def predict_cell_health(cell_data):
    # Load models
    damage_model, damage_scaler, backup_model, backup_scaler = load_models()
    
    # Prepare features
    X = prepare_features(cell_data)
    
    # Scale features
    X_scaled = damage_scaler.transform(X)
    
    # Predict damage probability
    damage_prob = damage_model.predict_proba(X_scaled)[:, 1]
    
    # Predict required backups
    X_scaled_backup = backup_scaler.transform(X)
    backup_needed = backup_model.predict(X_scaled_backup)
    
    return damage_prob, backup_needed

if __name__ == '__main__':
    # Example usage
    sample_data = pd.DataFrame({
        'voltage': [3.8],
        'current': [15.0],
        'temperature': [30.0],
        'soc': [75.0]
    })
    
    damage_prob, backup_needed = predict_cell_health(sample_data)
    print(f"Probability of cell damage: {damage_prob[0]:.2%}")
    print(f"Number of backup cells needed: {int(np.ceil(backup_needed[0]))}")
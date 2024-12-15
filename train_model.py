import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import os

def load_and_prepare_data():
    # Load the cell data
    df = pd.read_csv('cell_data.csv')
    
    # Create features for cell damage prediction
    df['voltage_current_ratio'] = df['voltage'] / (df['current'].abs() + 1e-6)
    df['power'] = df['voltage'] * df['current']
    
    # Create target for backup prediction (assuming cells with status 1 need backup)
    df['needs_backup'] = df.groupby(['timestamp', 'group_name'])['status'].transform('sum')
    
    return df

def enhance_features(df):
    # Add time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Add engineering features
    df['voltage_gradient'] = df.groupby('cell_number')['voltage'].diff()
    df['temp_gradient'] = df.groupby('cell_number')['temperature'].diff()
    df['power_efficiency'] = df['power'] / (df['temperature'] + 273.15)  # Simple efficiency metric
    
    # Add statistical features
    df['voltage_rolling_mean'] = df.groupby('cell_number')['voltage'].rolling(window=3).mean().reset_index(0, drop=True)
    df['temp_rolling_std'] = df.groupby('cell_number')['temperature'].rolling(window=3).std().reset_index(0, drop=True)
    
    return df.fillna(0)  # Fill NaN values created by diff and rolling operations

def train_damage_prediction_model(X, y):
    # Initialize scaler and model
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump((model, scaler), 'battery_damage_model.pkl')
    print("Model saved as 'battery_damage_model.pkl'")
    
    return model, scaler

def train_backup_prediction_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = regr.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("\nBackup Prediction MSE:", mse)
    
    return regr, scaler

def save_models(damage_model, damage_scaler, backup_model, backup_scaler):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save models and scalers
    joblib.dump(damage_model, 'models/damage_model.pkl')
    joblib.dump(damage_scaler, 'models/damage_scaler.pkl')
    joblib.dump(backup_model, 'models/backup_model.pkl')
    joblib.dump(backup_scaler, 'models/backup_scaler.pkl')

if __name__ == '__main__':
    # Load and prepare data
    df = load_and_prepare_data()
    df = enhance_features(df)
    
    # Define enhanced feature set
    features = ['voltage', 'current', 'temperature', 'soc', 
               'voltage_current_ratio', 'power', 'hour', 'day_of_week',
               'voltage_gradient', 'temp_gradient', 'power_efficiency',
               'voltage_rolling_mean', 'temp_rolling_std']
    
    X_damage = df[features]
    y_damage = df['status']
    
    X_backup = df[features]
    y_backup = df['needs_backup']
    
    # Train and save models
    damage_model, damage_scaler = train_damage_prediction_model(X_damage, y_damage)
    backup_model, backup_scaler = train_backup_prediction_model(X_backup, y_backup)
    save_models(damage_model, damage_scaler, backup_model, backup_scaler)
    
    print("\nModels saved successfully!")
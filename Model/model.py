import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE

class BatteryFaultDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def extract_features(self, cell_data):
        """Extract features from battery cell data."""
        voltage = cell_data['voltage'].values
        temp = cell_data['temperature'].values
        time = cell_data['time'].values
        
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

    def prepare_data(self, data):
        """Prepare features and labels from battery data."""
        features = []
        labels = []
        
        for cell_id in data['cell_id'].unique():
            cell_data = data[data['cell_id'] == cell_id]
            features.append(self.extract_features(cell_data))
            labels.append(cell_data['fault'].iloc[0])
            
        return np.array(features), np.array(labels)

    def train(self, data_path):
        """Train the fault detection model."""
        # Load data
        data = pd.read_csv(data_path)
        
        # Prepare features and labels
        X, y = self.prepare_data(data)
        
        # Balance dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
    def predict(self, cell_data):
        """Predict fault for new battery cell data."""
        features = self.extract_features(cell_data)
        return self.model.predict(features.reshape(1, -1))[0]
    
    def save_model(self, filename='battery_fault_detector.pkl'):
        """Save trained model."""
        joblib.dump(self.model, filename)
        print(f"Model saved as '{filename}'")
    
    @classmethod
    def load_model(cls, filename='battery_fault_detector.pkl'):
        """Load trained model."""
        detector = cls()
        detector.model = joblib.load(filename)
        return detector

if __name__ == "__main__":
    # Train model
    detector = BatteryFaultDetector()
    detector.train('battery_data.csv')
    detector.save_model()
    
    # Example prediction
    test_data = pd.DataFrame({
        'time': np.linspace(0, 3600, 1000),
        'voltage': 4.2 - 0.5 * np.random.random(1000),
        'temperature': 25 + np.random.normal(0, 0.5, 1000),
    })
    
    prediction = detector.predict(test_data)
    print(f"\nTest prediction: {'Fault' if prediction == 1 else 'Healthy'}")
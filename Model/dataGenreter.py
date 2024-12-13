import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_battery_data(num_cells=100, samples_per_cell=1000, save_csv=True):
    """Generate realistic battery cell voltage profiles with various conditions"""
    
    # Base parameters
    nominal_voltage = 4.2  # Full charge voltage
    end_voltage = 2.5     # Cut-off voltage
    temp_nominal = 25     # Nominal temperature (°C)
    
    all_data = []
    
    for cell_id in range(num_cells):
        # Random parameters for this cell
        capacity_factor = np.random.normal(1.0, 0.05)  # Cell-to-cell variation
        temp_factor = np.random.normal(temp_nominal, 3)  # Temperature variation
        
        # Time points
        time = np.linspace(0, 3600, samples_per_cell)  # 1 hour discharge
        
        # Generate base discharge curve
        voltage = nominal_voltage - (nominal_voltage - end_voltage) * (
            1 - np.exp(-3 * time/3600)
        ) * capacity_factor
        
        # Add noise and artifacts
        noise = np.random.normal(0, 0.01, samples_per_cell)
        voltage += noise
        
        # Determine if this cell will have a fault
        has_fault = np.random.random() < 0.2  # 20% fault rate
        
        if has_fault:
            fault_type = np.random.choice(['sudden_drop', 'gradual_decline', 'oscillation'])
            fault_start = np.random.randint(samples_per_cell // 3, samples_per_cell)
            
            if fault_type == 'sudden_drop':
                voltage[fault_start:] -= 0.5
            elif fault_type == 'gradual_decline':
                decline = np.linspace(0, 0.8, samples_per_cell - fault_start)
                voltage[fault_start:] -= decline
            else:  # oscillation
                oscillation = 0.2 * np.sin(np.linspace(0, 10*np.pi, samples_per_cell - fault_start))
                voltage[fault_start:] += oscillation
        
        # Temperature effects
        temp = temp_factor + np.random.normal(0, 0.5, samples_per_cell)
        
        # Create DataFrame for this cell
        df = pd.DataFrame({
            'cell_id': cell_id,
            'time': time,
            'voltage': voltage,
            'temperature': temp,
            'fault': int(has_fault),
            'fault_start': fault_start if has_fault else -1,
            'fault_type': fault_type if has_fault else 'none'
        })
        
        all_data.append(df)
    
    # Combine all cells' data
    final_df = pd.concat(all_data, ignore_index=True)
    
    if save_csv:
        final_df.to_csv('battery_data.csv', index=False)
        print(f"Generated data for {num_cells} cells saved to battery_data.csv")
    
    return final_df

# Generate data
data = generate_battery_data(num_cells=100)

# Plot some example cells
plt.figure(figsize=(12, 6))
for cell_id in range(5):
    cell_data = data[data['cell_id'] == cell_id]
    plt.plot(cell_data['time'], cell_data['voltage'], 
             label=f"Cell {cell_id} (Fault: {cell_data['fault'].iloc[0]})")

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Example Battery Cell Discharge Profiles')
plt.legend()
plt.grid(True)
plt.show()
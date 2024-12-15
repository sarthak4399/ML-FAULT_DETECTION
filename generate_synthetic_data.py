import random
from datetime import datetime, timedelta
import json
import csv

def generate_cell_data():
    voltage = round(random.uniform(3.0, 4.2), 3)  # typical Li-ion cell voltage
    soc = ((voltage - 3.0) / (4.2 - 3.0)) * 100  # approximate SOC calculation
    
    return {
        'current': round(random.uniform(-20, 20), 3),  # charging/discharging current
        'temperature': round(random.uniform(15, 45), 2),  # safe operating temperature
        'voltage': voltage,
        'soc': round(soc, 2),  # state of charge
        'status': random.choice([0, 1])  # 0: normal, 1: fault
    }

def generate_group_data():
    # Generate more correlated cell voltages within a group
    base_voltage = round(random.uniform(3.0, 4.2), 3)
    cell_voltages = [
        round(base_voltage + random.uniform(-0.1, 0.1), 3) 
        for _ in range(3)
    ]
    total_voltage = round(sum(cell_voltages), 3)
    
    return {
        'voltage': total_voltage,
        'cell1': generate_cell_data(),
        'cell2': generate_cell_data(),
        'cell3': generate_cell_data(),
        'connect_state': random.choice([True, False])
    }

def generate_timestamp(base_time, index):
    return (base_time + timedelta(seconds=index)).strftime("%Y-%m-%d %H:%M:%S")

def generate_synthetic_dataset(num_samples=100):
    base_time = datetime.now()
    dataset = {}
    
    for i in range(num_samples):
        timestamp = generate_timestamp(base_time, i)
        dataset[timestamp] = {
            'group1': generate_group_data(),
            'group2': generate_group_data(),
            'group3': generate_group_data()
        }
    
    return dataset

def flatten_data_for_groups(dataset):
    rows = []
    for timestamp, data in dataset.items():
        for group_name, group_data in data.items():
            row = {
                'timestamp': timestamp,
                'group_name': group_name,
                'group_voltage': group_data['voltage'],
                'connect_state': group_data['connect_state']
            }
            rows.append(row)
    return rows

def flatten_data_for_cells(dataset):
    rows = []
    for timestamp, data in dataset.items():
        for group_name, group_data in data.items():
            for cell_num in [1, 2, 3]:
                cell_data = group_data[f'cell{cell_num}']
                row = {
                    'timestamp': timestamp,
                    'group_name': group_name,
                    'cell_number': cell_num,
                    'current': cell_data['current'],
                    'temperature': cell_data['temperature'],
                    'voltage': cell_data['voltage'],
                    'soc': cell_data['soc'],
                    'status': cell_data['status']
                }
                rows.append(row)
    return rows

def save_to_csv(data, filename, fieldnames):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    # Generate 100 samples
    synthetic_data = generate_synthetic_dataset(100)
    
    # Prepare and save group-level data
    group_data = flatten_data_for_groups(synthetic_data)
    group_fields = ['timestamp', 'group_name', 'group_voltage', 'connect_state']
    save_to_csv(group_data, 'group_data.csv', group_fields)
    
    # Prepare and save cell-level data
    cell_data = flatten_data_for_cells(synthetic_data)
    cell_fields = ['timestamp', 'group_name', 'cell_number', 'current', 
                  'temperature', 'voltage', 'soc', 'status']
    save_to_csv(cell_data, 'cell_data.csv', cell_fields)
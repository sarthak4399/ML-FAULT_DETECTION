from flask import Flask, request, jsonify
from flask_cors import CORS	
from datetime import datetime
import logging
import json
import csv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})
# CSV file configuration
CSV_FILE = 'LAST_FINAL.csv'
CSV_HEADERS = [
    'timestamp', 'group_id', 'group_voltage', 'connect_state',
    'cell_id', 'cell_voltage', 'cell_current', 'cell_temperature', 'cell_status'
]

def setup_csv():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def save_to_csv(data, timestamp):
    """Save battery data to CSV."""
    rows = []
    for group_id in range(1, 4):
        group_key = f'group{group_id}'
        group_data = data.get(group_key, {})
        
        for cell_id in range(1, 4):
            cell_key = f'cell{cell_id}'
            cell = group_data.get(cell_key, {})
            
            row = [
                timestamp,
                group_id,
                group_data.get('voltage', 0),
                int(group_data.get('connect_state', False)),
                cell_id,
                cell.get('voltage', 0),
                cell.get('current', 0),
                cell.get('temperature', 0),
                cell.get('status', 0)
            ]
            rows.append(row)
    
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

@app.route('/data', methods=['POST'])
def receive_data():
    try:
        # Get data
        data = request.get_json()
        if not data:
            raise ValueError("No data received")
            
        # Log received data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"\nReceived data at {timestamp}:")
        logging.info(json.dumps(data, indent=2))
        
        # Ensure CSV exists
        setup_csv()
        
        # Save data
        save_to_csv(data, timestamp)
        
        return jsonify({
            'status': 'success',
            'message': 'Data saved successfully',
            'timestamp': timestamp
        })
        
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
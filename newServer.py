# server.py
from flask import Flask, request, jsonify
from datetime import datetime
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_data():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Print received data with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Received data:")
        print(json.dumps(data, indent=2))
        
        # Validate data structure
        for group_key in data:
            if not group_key.startswith('group'):
                continue
                
            group_data = data[group_key]
            print(f"\nProcessing {group_key}:")
            print(f"Voltage: {group_data.get('voltage')}")
            print(f"Connection state: {group_data.get('connect_state')}")
            
            # Process cell data
            for cell_key in group_data:
                if not cell_key.startswith('cell'):
                    continue
                    
                cell_data = group_data[cell_key]
                print(f"\n{cell_key}:")
                print(f"Current: {cell_data.get('current')}A")
                print(f"Temperature: {cell_data.get('temperature')}°C")
                print(f"Voltage: {cell_data.get('voltage')}V")
                print(f"Status: {cell_data.get('status')}")
        
        return jsonify({
            "status": "success",
            "message": "Data received successfully",
            "timestamp": timestamp
        }), 200
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    print("Starting server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)
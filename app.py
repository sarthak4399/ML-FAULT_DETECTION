# # app.py
# from flask import Flask, request, jsonify
# from model import BatteryLifetimePredictor
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load the trained model
# predictor = BatteryLifetimePredictor.load_model('battery_predictor.pkl')

# @app.route('/predict', methods=['POST'])

# def predict():
#     try:
#         data = request.get_json()
#         # Extract features from request data
#         cell_data = {
#             'voltage_mean': np.mean([cell['voltage'] for cell in data['cells']]),
#             'voltage_std': np.std([cell['voltage'] for cell in data['cells']]),
#             'temperature_mean': np.mean([cell['temperature'] for cell in data['cells']]),
#             'temperature_max': max([cell['temperature'] for cell in data['cells']]),
#             'voltage_trend': data.get('voltage_trend', 0),
#             'temp_stress': len([c for c in data['cells'] if c['temperature'] > 40]) / len(data['cells']),
#             'voltage_stability': np.std([cell['voltage'] for cell in data['cells']]),
#             'time_span': data.get('time_span', 0)
#         }
        
#         prediction = predictor.predict(cell_data)
#         return jsonify(prediction)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)


# app.py
from flask import Flask, request, jsonify
from model import BatteryLifetimePredictor
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
predictor = BatteryLifetimePredictor.load_model('battery_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        predictions = []
        
        # Process each group
        for group_id in range(1, 4):
            group_key = f'group{group_id}'
            group_data = data.get(group_key, {})
            
            # Collect cell data from group
            cells = []
            for cell_id in range(1, 4):
                cell_key = f'cell{cell_id}'
                if cell_key in group_data:
                    cells.append(group_data[cell_key])
            
            if cells:
                # Extract features
                cell_data = {
                    'voltage_mean': np.mean([cell['voltage'] for cell in cells]),
                    'voltage_std': np.std([cell['voltage'] for cell in cells]),
                    'temperature_mean': np.mean([cell['temperature'] for cell in cells]),
                    'temperature_max': max([cell['temperature'] for cell in cells]),
                    'voltage_trend': group_data.get('voltage_trend', 0),
                    'temp_stress': len([c for c in cells if c['temperature'] > 40]) / len(cells),
                    'voltage_stability': np.std([cell['voltage'] for cell in cells]),
                    'time_span': group_data.get('time_span', 0)
                }
                
                # Get prediction for this group
                prediction = predictor.predict(cell_data)
                predictions.append({
                    'group_id': group_id,
                    'group_voltage': group_data.get('voltage', 0),
                    'connect_state': group_data.get('connect_state', False),
                    'prediction': prediction
                })
                print(predictions)
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        

        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
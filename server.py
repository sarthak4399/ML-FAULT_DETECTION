from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import csv
import os
from datetime import datetime

class BatteryDataHandler(BaseHTTPRequestHandler):
    csv_file = 'battery_data.csv'
    
    @staticmethod
    def setup_csv():
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(BatteryDataHandler.csv_file):
            with open(BatteryDataHandler.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'timestamp',
                    'group_id',
                    'group_voltage',
                    'connect_state',
                    'cell_id',
                    'cell_voltage',
                    'cell_current',
                    'cell_temperature',
                    'cell_status'
                ]
                writer.writerow(headers)

    def save_to_csv(self, data, timestamp):
        """Save battery data to CSV file."""
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
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def do_POST(self):
        """Handle POST requests with battery data."""
        try:
            # Read request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Ensure CSV exists
            self.setup_csv()
            
            # Save data
            self.save_to_csv(data, timestamp)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "success", "message": "Data saved"}
            self.wfile.write(json.dumps(response).encode())
            
            # Print received data
            print(f"\nReceived data at {timestamp}:")
            for group in range(1, 4):
                print(f"group{group}: {data.get(f'group{group}', {})}")
            
        except Exception as e:
			
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "error", "message": str(e)}
            self.wfile.write(json.dumps(response).encode())

def run_server(port=8000):
    """Run the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, BatteryDataHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
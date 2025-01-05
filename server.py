from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class EnvAwareHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Handle .env request
        if self.path == '/.env':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get environment variables
            env_vars = {
                'FACE_API_KEY': os.getenv('FACE_API_KEY', ''),
                'FACE_API_SECRET': os.getenv('FACE_API_SECRET', '')
            }
            
            self.wfile.write(json.dumps(env_vars).encode())
            return
        
        # For all other paths, serve files as usual
        return SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        # Add CORS headers for all responses
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, EnvAwareHandler)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 
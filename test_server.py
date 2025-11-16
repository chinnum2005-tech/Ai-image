from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import traceback
import sys

app = Flask(__name__)

# Configure CORS to allow all origins for development
CORS(app)  # This allows all origins in development

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({
        'status': 'success',
        'message': 'Backend is connected!',
        'data': {
            'backend': 'Flask',
            'status': 'running',
            'version': '1.0'
        }
    })

@app.route('/api/upload_and_analyze', methods=['POST'])
def mock_upload():
    try:
        print("\n=== Received upload request ===")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request form data: {request.form}")
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            print("Error: No file part in request")
            return jsonify({
                'status': 'error',
                'message': 'No file part in request',
                'details': 'Please include an image file in your request'
            }), 400
        
        # Get the file
        file = request.files['image']
        
        if file.filename == '':
            print("Error: No selected file")
            return jsonify({
                'status': 'error',
                'message': 'No selected file',
                'details': 'Please select a file to upload'
            }), 400
        
        print(f"Processing file: {file.filename}, Content-Type: {file.content_type}")
        
        # Read the file content first
        try:
            file_content = file.read()
            file_size_kb = len(file_content) / 1024
            print(f"File size: {file_size_kb:.2f} KB")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error reading file',
                'details': str(e)
            }), 400
        
        # Create mock response
        response_data = {
        'status': 'success',
        'prediction': {
            'is_ai_generated': True,
            'confidence': 0.85,
            'class': 'AI-Generated',
            'message': 'This is a mock response. The model is not loaded.'
        },
        'image_info': {
            'filename': file.filename,
            'size_kb': file_size_kb,
            'mimetype': file.mimetype,
            'width': 1024,
            'height': 1024,
            'format': file.filename.split('.')[-1].upper()
        },
        'features': {
            'anomaly_score': 0.15,
            'confidence_interval': [0.8, 0.9],
            'key_indicators': {
                'color_consistency': 0.92,
                'noise_pattern': 0.87,
                'edge_clarity': 0.45
            }
        },
        'analysis': {
            'risk_level': 'HIGH',
            'details': 'The image shows several indicators of AI generation.',
            'recommendations': [
                'Verify the source of this image.',
                'Cross-reference with other sources if possible.'
            ]
        },
        'timestamp': '2025-11-15T13:30:00Z'
    }
    
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = 5008
    print(f"=== Starting Test Server ===")
    print(f"Server will run on: http://localhost:{port}")
    print(f"Test endpoint: GET http://localhost:{port}/api/test")
    print(f"Mock upload endpoint: POST http://localhost:{port}/api/upload_and_analyze")
    print("\n=== Server Logs ===")
    try:
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
    except Exception as e:
        print(f"\n=== ERROR: Failed to start server on port {port} ===")
        print(f"Error: {str(e)}")
        print("\nCommon solutions:")
        print(f"1. Try a different port by changing 'port = {port}' to another number (e.g., 5009)")
        print("2. Check if another program is using this port")
        print("3. Run as administrator in case of permission issues")

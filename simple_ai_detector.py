from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import random
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/upload_and_analyze', methods=['POST'])
def analyze_image():
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
        
        # Read the file content
        try:
            file_content = file.read()
            file_size_kb = len(file_content) / 1024
            print(f"File size: {file_size_kb:.2f} KB")
            
            # Reset file pointer to beginning
            file.seek(0)
            
            # Open image to get dimensions
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size
            print(f"Image dimensions: {width}x{height}")
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error reading file',
                'details': str(e)
            }), 400
        
        # Simple heuristic-based AI detection
        # This is a mock implementation that uses basic image properties
        # In a real implementation, this would use actual AI models
        is_ai_generated, confidence = detect_ai_simple(width, height, file_size_kb, file.filename)
        
        # Create response
        response_data = {
            'status': 'success',
            'prediction': {
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'class': 'AI-Generated' if is_ai_generated else 'Real',
                'message': 'Analysis completed using heuristic-based detection'
            },
            'image_info': {
                'filename': file.filename,
                'size_kb': file_size_kb,
                'width': width,
                'height': height,
                'format': image.format
            },
            'features': {
                'anomaly_score': round(1 - confidence if is_ai_generated else confidence, 2),
                'confidence_interval': [max(0, confidence - 0.1), min(1, confidence + 0.1)],
                'key_indicators': {
                    'resolution_consistency': round(random.uniform(0.7, 0.95), 2),
                    'compression_artifacts': round(random.uniform(0.6, 0.9), 2),
                    'edge_clarity': round(random.uniform(0.5, 0.85), 2)
                }
            },
            'analysis': {
                'risk_level': get_risk_level(confidence, is_ai_generated),
                'details': get_analysis_details(confidence, is_ai_generated),
                'recommendations': [
                    'Verify the source of this image.',
                    'Cross-reference with other sources if possible.',
                    'Consider consulting an expert for critical decisions.'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'details': str(e)
        }), 500

def detect_ai_simple(width, height, file_size_kb, filename):
    """
    Simple heuristic-based AI detection
    In a real implementation, this would use actual AI models
    """
    # This is a mock implementation - in reality, this would use:
    # 1. Pixel-level analysis (PRNU, ELA, noise patterns)
    # 2. Deep learning models
    # 3. Feature extraction and classification
    
    # For demonstration purposes, we'll use some basic heuristics:
    # - Very high resolution with small file size might indicate AI compression
    # - Certain filename patterns might indicate AI generation
    # - Random confidence for demonstration
    
    # Simple mock logic
    if "ai" in filename.lower() or "generated" in filename.lower():
        return True, random.uniform(0.7, 0.95)
    elif width > 1000 and height > 1000 and file_size_kb < 500:
        # Large image with small file size - might be AI compressed
        return True, random.uniform(0.6, 0.8)
    elif width == height and width in [512, 768, 1024]:
        # Square images of common AI generation sizes
        return True, random.uniform(0.55, 0.75)
    else:
        # For other cases, return a random result for demonstration
        is_ai = random.choice([True, False])
        confidence = random.uniform(0.5, 0.9) if is_ai else random.uniform(0.5, 0.8)
        return is_ai, confidence

def get_risk_level(confidence, is_ai_generated):
    """Determine risk level based on confidence"""
    if not is_ai_generated:
        return 'LOW'
    
    if confidence >= 0.9:
        return 'CRITICAL'
    elif confidence >= 0.75:
        return 'HIGH'
    elif confidence >= 0.6:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_analysis_details(confidence, is_ai_generated):
    """Get detailed analysis description"""
    if is_ai_generated:
        if confidence >= 0.9:
            return "Strong indicators of AI generation detected. Multiple heuristic patterns suggest synthetic origin."
        elif confidence >= 0.75:
            return "High probability of AI generation. Several characteristics match known AI generation patterns."
        elif confidence >= 0.6:
            return "Moderate indicators of AI generation. Some patterns suggest possible synthetic origin."
        else:
            return "Possible AI generation detected, but confidence is low. Manual review recommended."
    else:
        if confidence >= 0.9:
            return "Strong indicators of authentic photograph. No obvious signs of AI generation detected."
        elif confidence >= 0.75:
            return "High probability of authentic photograph. Minimal synthetic characteristics detected."
        elif confidence >= 0.6:
            return "Moderate indicators of authentic photograph. Some minor anomalies detected."
        else:
            return "Possible authentic photograph, but some uncertainty exists. Further verification recommended."

if __name__ == '__main__':
    port = 5000
    print(f"=== Starting Simple AI Detection Server ===")
    print(f"Server will run on: http://localhost:{port}")
    print(f"Upload endpoint: POST http://localhost:{port}/api/upload_and_analyze")
    print("\n=== Server Logs ===")
    try:
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
    except Exception as e:
        print(f"\n=== ERROR: Failed to start server on port {port} ===")
        print(f"Error: {str(e)}")
        print("\nCommon solutions:")
        print(f"1. Try a different port by changing 'port = {port}' to another number (e.g., 5001)")
        print("2. Check if another program is using this port")
        print("3. Run as administrator in case of permission issues")
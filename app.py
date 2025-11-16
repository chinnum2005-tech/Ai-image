"""
Flask Backend API for AI-Generated Image Detection
Provides REST endpoints for image analysis
"""

import os
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import logging

# Import actual modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import ImagePreprocessor
from feature_extraction import PixelWiseFeatureExtractor
from model import AIImageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow requests from the frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:4174",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:4174"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SECRET_KEY'] = 'ai-detection-secret-key-2024'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global model and preprocessor
model = None
preprocessor = None
feature_extractor = None
model_type = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model"""
    global model, preprocessor, feature_extractor, model_type
    
    try:
        # Load model metadata
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
                model_type = metadata.get('model_type', 'cnn')
        else:
            model_type = 'cnn'
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor()
        
        # Initialize feature extractor for hybrid model
        if model_type == 'hybrid':
            feature_extractor = PixelWiseFeatureExtractor()
        
        # Load model
        if os.path.exists('model.h5'):
            logger.info("Loading trained model...")
            model = tf.keras.models.load_model('model.h5')
            logger.info(f"Model loaded successfully! Type: {model_type}")
        else:
            logger.warning("No trained model found. Creating default model...")
            # Create a default model
            if model_type == 'hybrid':
                detector = AIImageDetector()
                model = detector.build_hybrid_model()
            else:
                detector = AIImageDetector()
                model = detector.build_cnn_only_model()
            detector.compile_model()
            detector.save_model('model.h5')
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Create a fallback model
        detector = AIImageDetector()
        model = detector.build_cnn_only_model()
        detector.compile_model()


@app.route('/')
def index():
    """Serve a simple HTML interface"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Image Detection API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .endpoints {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
            }
            .endpoint {
                margin: 15px 0;
                padding: 10px;
                background: white;
                border-left: 4px solid #667eea;
                border-radius: 3px;
            }
            code {
                background: #f0f0f0;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            .method {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                color: white;
                font-weight: bold;
                margin-right: 10px;
                font-size: 12px;
            }
            .get { background: #28a745; }
            .post { background: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 AI-Generated Image Detection API</h1>
            <p style="text-align: center; color: #666;">
                Detect whether images are real or AI-generated using pixel-wise feature extraction
            </p>
            
            <div class="endpoints">
                <h3>Available Endpoints:</h3>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/health</code>
                    <p>Check API health status</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/api/analyze</code>
                    <p>Analyze an image for AI generation</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/api/analyze_batch</code>
                    <p>Analyze multiple images</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/features/<image_id></code>
                    <p>Get pixel-wise features visualization</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/model_info</code>
                    <p>Get model information and metrics</p>
                </div>
            </div>
            
            <p style="margin-top: 30px; text-align: center; color: #666;">
                Model Type: <strong>{{model_type}}</strong> | 
                Status: <strong style="color: #28a745;">Ready</strong>
            </p>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, model_type=model_type or 'Not loaded')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze a single image"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename or 'unnamed')
        upload_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(filepath)
        
        # Preprocess image
        if preprocessor is None:
            return jsonify({'error': 'Preprocessor not initialized'}), 500
        img = preprocessor.prepare_for_cnn(filepath)
        
        # Extract pixel features if using hybrid model
        if model_type == 'hybrid' and feature_extractor:
            pixel_features = feature_extractor.get_feature_vector(filepath)
            
            # Normalize features
            if os.path.exists('feature_mean.npy') and os.path.exists('feature_std.npy') and preprocessor is not None:
                mean = np.load('feature_mean.npy')
                std = np.load('feature_std.npy')
                pixel_features = preprocessor.apply_normalization(
                    pixel_features.reshape(1, -1), mean, std
                )
            else:
                pixel_features = pixel_features.reshape(1, -1)
            
            # Make prediction
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
            prediction = model.predict([img, pixel_features], verbose=0)[0][0]
        else:
            # CNN-only prediction
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
            prediction = model.predict(img, verbose=0)[0][0]
        
        # Determine result - temporary inversion until model retrained
        is_ai_generated = prediction <= 0.5
        confidence = float(1 - prediction) if is_ai_generated else float(prediction)
        
        # Generate feature visualizations
        feature_maps = None
        if feature_extractor:
            try:
                feature_maps = feature_extractor.get_feature_maps(filepath)
                
                # Save feature maps
                if feature_maps:
                    # Save PRNU map
                    prnu_path = os.path.join(app.config['RESULTS_FOLDER'], f"{upload_id}_prnu.png")
                    cv2.imwrite(prnu_path, feature_maps['prnu'])
                    
                    # Save ELA map
                    ela_path = os.path.join(app.config['RESULTS_FOLDER'], f"{upload_id}_ela.png")
                    cv2.imwrite(ela_path, cv2.cvtColor(feature_maps['ela'], cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.error(f"Error generating feature maps: {e}")
        
        # Prepare response
        result = {
            'id': upload_id,
            'filename': filename,
            'is_ai_generated': bool(is_ai_generated),
            'confidence': confidence,
            'prediction_score': float(prediction),
            'result': 'AI-Generated' if is_ai_generated else 'Real',
            'analysis': {
                'risk_level': get_risk_level(confidence, is_ai_generated),
                'details': get_analysis_details(confidence, is_ai_generated)
            },
            'feature_maps_available': feature_maps is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple images"""
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename or 'unnamed')
                upload_id = str(uuid.uuid4())
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
                file.save(filepath)
                
                try:
                    # Preprocess image
                    if preprocessor is None:
                        results.append({
                            'filename': filename,
                            'error': 'Preprocessor not initialized'
                        })
                        continue
                    img = preprocessor.prepare_for_cnn(filepath)
                    
                    # Make prediction
                    if model_type == 'hybrid' and feature_extractor:
                        pixel_features = feature_extractor.get_feature_vector(filepath)
                        if os.path.exists('feature_mean.npy') and preprocessor is not None:
                            mean = np.load('feature_mean.npy')
                            std = np.load('feature_std.npy')
                            pixel_features = preprocessor.apply_normalization(
                                pixel_features.reshape(1, -1), mean, std
                            )
                        else:
                            pixel_features = pixel_features.reshape(1, -1)
                        if model is None:
                            results.append({
                                'filename': filename,
                                'error': 'Model not initialized'
                            })
                            continue
                        prediction = model.predict([img, pixel_features], verbose=0)[0][0]
                    else:
                        if model is None:
                            results.append({
                                'filename': filename,
                                'error': 'Model not initialized'
                            })
                            continue
                        prediction = model.predict(img, verbose=0)[0][0]
                    
                    # Determine result - temporary inversion until model retrained
                    is_ai_generated = prediction <= 0.5
                    confidence = float(1 - prediction) if is_ai_generated else float(prediction)
                    
                    results.append({
                        'filename': filename,
                        'is_ai_generated': bool(is_ai_generated),
                        'confidence': confidence,
                        'result': 'AI-Generated' if is_ai_generated else 'Real'
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': filename,
                        'error': str(e)
                    })
                
                # Clean up
                os.remove(filepath)
        
        return jsonify({
            'total': len(files),
            'processed': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/features/<image_id>', methods=['GET'])
def get_feature_visualization(image_id):
    """Get feature visualization for an analyzed image"""
    try:
        # Check if PRNU map exists
        prnu_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_prnu.png")
        ela_path = os.path.join(app.config['RESULTS_FOLDER'], f"{image_id}_ela.png")
        
        feature_type = request.args.get('type', 'prnu')
        
        if feature_type == 'prnu' and os.path.exists(prnu_path):
            return send_file(prnu_path, mimetype='image/png')
        elif feature_type == 'ela' and os.path.exists(ela_path):
            return send_file(ela_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Feature visualization not found'}), 404
            
    except Exception as e:
        logger.error(f"Error getting feature visualization: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information and metrics"""
    try:
        info = {
            'model_type': model_type,
            'model_loaded': model is not None,
            'input_shape': [256, 256, 3],
            'features': {
                'pixel_features': model_type == 'hybrid',
                'cnn_features': True
            }
        }
        
        # Load model metadata if available
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
                info['training_date'] = metadata.get('training_date')
                info['test_results'] = metadata.get('test_results')
        
        # Model architecture info
        if model:
            info['total_parameters'] = int(model.count_params())
            info['layers'] = len(model.layers)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    """Combined upload and analyze endpoint with detailed response"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Save file
        filename = secure_filename(file.filename or 'unnamed')
        upload_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(filepath)
        
        # Read image for analysis
        img_orig = cv2.imread(filepath)
        if img_orig is None:
            return jsonify({'error': 'Could not read image file'}), 400
        height, width = img_orig.shape[:2]
        
        # Preprocess for model
        if preprocessor is None:
            return jsonify({'error': 'Preprocessor not initialized'}), 500
        img = preprocessor.prepare_for_cnn(filepath)
        
        # Extract features and make prediction
        pixel_features_data = {}
        
        if model_type == 'hybrid' and feature_extractor:
            # Extract all features
            all_features = feature_extractor.extract_all_features(filepath)
            pixel_features_data = all_features
            
            # Get feature vector
            pixel_features = feature_extractor.get_feature_vector(filepath)
            
            # Normalize
            if os.path.exists('feature_mean.npy') and preprocessor is not None:
                mean = np.load('feature_mean.npy')
                std = np.load('feature_std.npy')
                pixel_features = preprocessor.apply_normalization(
                    pixel_features.reshape(1, -1), mean, std
                )
            else:
                pixel_features = pixel_features.reshape(1, -1)
            
            # Predict
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
            prediction = model.predict([img, pixel_features], verbose=0)[0][0]
        else:
            # CNN-only
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
            prediction = model.predict(img, verbose=0)[0][0]
        
        # Generate visualizations
        visualizations = {}
        
        try:
            if feature_extractor:
                feature_maps = feature_extractor.get_feature_maps(filepath)
                
                # Convert to base64 for web display
                # PRNU map
                _, prnu_buffer = cv2.imencode('.png', feature_maps['prnu'])
                prnu_bytes = prnu_buffer.tobytes()
                prnu_base64 = base64.b64encode(prnu_bytes).decode('utf-8')
                visualizations['prnu'] = f"data:image/png;base64,{prnu_base64}"
                
                # ELA map
                ela_bgr = cv2.cvtColor(feature_maps['ela'], cv2.COLOR_RGB2BGR)
                _, ela_buffer = cv2.imencode('.png', ela_bgr)
                ela_bytes = ela_buffer.tobytes()
                ela_base64 = base64.b64encode(ela_bytes).decode('utf-8')
                visualizations['ela'] = f"data:image/png;base64,{ela_base64}"
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        # Determine result - temporary inversion until model retrained
        is_ai_generated = prediction <= 0.5
        confidence = float(1 - prediction) if is_ai_generated else float(prediction)
        
        # Prepare detailed response
        response = {
            'success': True,
            'id': upload_id,
            'filename': filename,
            'image_info': {
                'width': width,
                'height': height,
                'size_kb': os.path.getsize(filepath) / 1024
            },
            'prediction': {
                'is_ai_generated': bool(is_ai_generated),
                'confidence': confidence,
                'raw_score': float(prediction),
                'result_text': 'AI-Generated Image' if is_ai_generated else 'Real Photograph',
                'risk_level': get_risk_level(confidence, is_ai_generated)
            },
            'analysis': {
                'summary': get_analysis_details(confidence, is_ai_generated),
                'key_indicators': get_key_indicators(pixel_features_data) if pixel_features_data else [],
                'recommendations': get_recommendations(confidence, is_ai_generated)
            },
            'visualizations': visualizations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up
        # os.remove(filepath)  # Keep for potential future reference
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in upload and analyze: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
            return "Strong indicators of AI generation detected. Multiple pixel-level anomalies and synthesis patterns found."
        elif confidence >= 0.75:
            return "High probability of AI generation. Significant synthetic patterns detected in pixel analysis."
        elif confidence >= 0.6:
            return "Moderate indicators of AI generation. Some synthetic characteristics detected."
        else:
            return "Possible AI generation detected, but confidence is low. Manual review recommended."
    else:
        if confidence >= 0.9:
            return "Strong indicators of authentic photography. Consistent sensor noise and natural pixel patterns detected."
        elif confidence >= 0.75:
            return "High probability of real photograph. Natural image characteristics dominant."
        elif confidence >= 0.6:
            return "Likely a real photograph. Most pixel-level features consistent with camera capture."
        else:
            return "Possibly a real photograph, but some anomalies detected. Further analysis may be needed."


def get_key_indicators(features):
    """Extract key indicators from pixel features"""
    indicators = []
    
    if not features:
        return indicators
    
    # Check PRNU consistency
    if 'prnu_consistency' in features:
        if features['prnu_consistency'] < 0.3:
            indicators.append("Consistent sensor noise pattern (authentic)")
        elif features['prnu_consistency'] > 0.7:
            indicators.append("Inconsistent noise pattern (synthetic)")
    
    # Check ELA
    if 'ela_entropy' in features:
        if features['ela_entropy'] > 5.5:
            indicators.append("High compression artifact diversity")
        elif features['ela_entropy'] < 3.5:
            indicators.append("Uniform compression patterns")
    
    # Check correlation
    if 'correlation_variance' in features:
        if features['correlation_variance'] < 0.001:
            indicators.append("Unusual pixel correlation uniformity")
    
    # Check frequency
    if 'freq_ratio' in features:
        if features['freq_ratio'] > 100:
            indicators.append("Abnormal frequency distribution")
    
    return indicators[:5]  # Limit to 5 key indicators


def get_recommendations(confidence, is_ai_generated):
    """Get recommendations based on analysis"""
    recs = []
    
    if is_ai_generated:
        if confidence >= 0.75:
            recs.append("This image shows strong signs of AI generation")
            recs.append("Exercise caution if using for verification purposes")
            recs.append("Consider requesting original source or metadata")
        else:
            recs.append("Possible AI generation, but not conclusive")
            recs.append("Additional verification methods recommended")
    else:
        if confidence >= 0.75:
            recs.append("This appears to be an authentic photograph")
            recs.append("Pixel patterns consistent with camera sensors")
        else:
            recs.append("Likely authentic, but some anomalies present")
            recs.append("May be heavily edited or processed")
    
    return recs


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    logger.info("Starting AI Image Detection API...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Results folder: {app.config['RESULTS_FOLDER']}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

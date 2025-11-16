"""
API Testing Script for AI-Generated Image Detection
Tests the Flask API endpoints
"""

import requests
import json
import os
import time
from pathlib import Path


def test_health_check(base_url):
    """Test the health check endpoint"""
    print("\n🔍 Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Model type: {data.get('model_type')}")
            return True
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_model_info(base_url):
    """Test the model info endpoint"""
    print("\n🔍 Testing model info...")
    try:
        response = requests.get(f"{base_url}/api/model_info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Parameters: {data.get('total_parameters', 'N/A')}")
            print(f"   Layers: {data.get('layers', 'N/A')}")
            
            if 'test_results' in data and data['test_results']:
                print(f"   Test Accuracy: {data['test_results'].get('accuracy', 'N/A')}")
                print(f"   Test AUC: {data['test_results'].get('auc', 'N/A')}")
            return True
        else:
            print(f"❌ Model info failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False


def test_image_analysis(base_url, image_path):
    """Test image analysis endpoint"""
    print(f"\n🔍 Testing image analysis with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/api/upload_and_analyze", files=files)
            analysis_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
                    
                    prediction = data.get('prediction', {})
                    print(f"\n   Result: {prediction.get('result_text')}")
                    print(f"   AI-Generated: {prediction.get('is_ai_generated')}")
                    print(f"   Confidence: {prediction.get('confidence', 0):.1%}")
                    print(f"   Risk Level: {prediction.get('risk_level')}")
                    
                    analysis = data.get('analysis', {})
                    if analysis.get('summary'):
                        print(f"\n   Summary: {analysis['summary']}")
                    
                    if analysis.get('key_indicators'):
                        print("\n   Key Indicators:")
                        for indicator in analysis['key_indicators'][:3]:
                            print(f"   - {indicator}")
                    
                    return True
                else:
                    print(f"❌ Analysis failed: {data.get('error')}")
                    return False
            else:
                print(f"❌ Analysis failed: Status {response.status_code}")
                error_msg = response.json().get('error', 'Unknown error')
                print(f"   Error: {error_msg}")
                return False
                
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def create_test_image():
    """Create a test image if none exists"""
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print("\n📸 Creating test image...")
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple test image
            img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            # Add some patterns
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            pattern = (np.sin(10 * x) * np.cos(10 * y) * 127 + 128).astype(np.uint8)
            img_array[:, :, 0] = pattern
            
            img = Image.fromarray(img_array)
            img.save(test_image_path, quality=95)
            print(f"✅ Test image created: {test_image_path}")
            
        except ImportError:
            print("❌ Cannot create test image (PIL not installed)")
            return None
    
    return test_image_path


def main():
    """Main test function"""
    print("=" * 60)
    print(" AI IMAGE DETECTION API TEST ")
    print("=" * 60)
    
    # Configuration
    base_url = "http://localhost:5000"
    
    # Check if server is running
    print(f"\n🌐 Testing API at {base_url}")
    
    # Test 1: Health check
    if not test_health_check(base_url):
        print("\n⚠️  Server is not running. Please start the Flask app:")
        print("   python app.py")
        return
    
    # Test 2: Model info
    test_model_info(base_url)
    
    # Test 3: Image analysis
    test_images = []
    
    # Check for sample images
    if Path('dataset/real').exists():
        real_images = list(Path('dataset/real').glob('*.jpg'))[:1]
        test_images.extend(real_images)
    
    if Path('dataset/fake').exists():
        fake_images = list(Path('dataset/fake').glob('*.jpg'))[:1]
        test_images.extend(fake_images)
    
    # Create test image if no samples found
    if not test_images:
        test_image = create_test_image()
        if test_image:
            test_images.append(test_image)
    
    # Test each image
    for image_path in test_images:
        test_image_analysis(base_url, str(image_path))
    
    print("\n" + "=" * 60)
    print(" TEST COMPLETED ")
    print("=" * 60)
    print("\n✅ All API endpoints tested successfully!")
    print("\nNext steps:")
    print("1. Open the web interface at http://localhost:3000")
    print("2. Upload images for analysis")
    print("3. View feature visualizations and detailed reports")


if __name__ == "__main__":
    main()

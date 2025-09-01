#!/usr/bin/env python3
"""
Mr. Acne Detection API
Flask web service for acne detection using Roboflow AI

Deploy to Railway.app with GitHub integration
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import requests
import base64
import json
from datetime import datetime
import tempfile
import io
from PIL import Image
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'XDkVVhgoKR98eXBeMm15')

# Working AI model configuration
AI_MODEL = {
    "name": "facial-acne-detection-l06mq",
    "version": "1",
    "description": "Face-focused AI model",
    "confidence_threshold": 0.5
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image_file):
    """Validate uploaded image meets requirements"""
    try:
        # Open image to check dimensions
        image = Image.open(image_file)
        width, height = image.size
        
        # Check file size (in bytes)
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning
        
        # Validation rules
        min_pixels = 400 * 400  # 400x400 minimum
        max_file_size = 10 * 1024 * 1024  # 10MB max
        
        issues = []
        
        if width * height < min_pixels:
            issues.append(f"Image too small: {width}x{height}. Minimum: 600x600 for best results")
        
        if file_size > max_file_size:
            issues.append(f"File too large: {file_size/1024/1024:.1f}MB. Maximum: 10MB")
        
        # Check format
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            issues.append(f"Unsupported format: {image.format}. Use JPEG or PNG")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "width": width,
            "height": height,
            "file_size_mb": file_size / 1024 / 1024,
            "format": image.format
        }
        
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Invalid image file: {str(e)}"],
            "width": 0,
            "height": 0,
            "file_size_mb": 0,
            "format": "unknown"
        }

def encode_image_for_api(image_file):
    """Convert uploaded image to base64 for Roboflow API"""
    try:
        image_file.seek(0)
        image_data = base64.b64encode(image_file.read()).decode('ascii')
        return image_data
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def analyze_with_roboflow(image_data):
    """Send image to Roboflow AI for acne detection"""
    url = f"https://detect.roboflow.com/{AI_MODEL['name']}/{AI_MODEL['version']}?api_key={API_KEY}&format=json"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, data=image_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            all_predictions = result.get('predictions', [])
            
            # Filter by confidence threshold
            good_predictions = [
                p for p in all_predictions 
                if p.get('confidence', 0) >= AI_MODEL['confidence_threshold']
            ]
            
            return {
                "success": True,
                "predictions": good_predictions,
                "model_used": AI_MODEL['name'],
                "total_found": len(good_predictions),
                "processing_time": result.get('time', 0)
            }
            
        elif response.status_code == 403:
            return {
                "success": False,
                "error": "API access denied",
                "error_code": "ACCESS_DENIED"
            }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code}",
                "error_code": "API_ERROR"
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout - please try again",
            "error_code": "TIMEOUT"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_code": "UNKNOWN_ERROR"
        }

def create_annotated_image(original_image_file, predictions):
    """Create annotated image with detection boxes"""
    try:
        # Convert to OpenCV format
        original_image_file.seek(0)
        image_array = np.frombuffer(original_image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        height, width = image.shape[:2]
        
        # Mr. Acne brand colors
        colors = [
            (66, 179, 124),   # Primary green (BGR format)
            (122, 122, 255),  # Coral pink
            (0, 255, 0),      # Bright green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
        ]
        
        for i, detection in enumerate(predictions):
            # Get detection data
            x_center = detection['x']
            y_center = detection['y']
            box_width = detection['width']
            box_height = detection['height']
            confidence = detection['confidence']
            acne_type = detection['class']
            
            # Calculate box coordinates
            x1 = int(x_center - box_width/2)
            y1 = int(y_center - box_height/2)
            x2 = int(x_center + box_width/2)
            y2 = int(y_center + box_height/2)
            
            # Use brand colors
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"#{i+1}: {acne_type.upper()} {confidence*100:.1f}%"
            
            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label
            label_y = y1 - 10 if y1 > text_h + 15 else y2 + text_h + 15
            
            # Draw label background
            cv2.rectangle(image, (x1, label_y-text_h-5), (x1+text_w+10, label_y+5), color, -1)
            
            # Draw label text
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(image, label, (x1+5, label_y), font, font_scale, text_color, font_thickness)
        
        # Add Mr. Acne branding
        title = f"MR. ACNE DETECTION - {len(predictions)} lesions found"
        cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        # Convert back to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Save to memory
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        return img_buffer
        
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return None

# API Routes

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Mr. Acne Detection API",
        "version": "1.0.0",
        "ai_model": AI_MODEL['name']
    })

@app.route('/detect', methods=['POST'])
def detect_acne():
    """Main acne detection endpoint"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided",
                "error_code": "NO_IMAGE"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image file selected",
                "error_code": "EMPTY_FILENAME"
            }), 400
        
        # Validate image
        validation = validate_image(image_file)
        if not validation['valid']:
            return jsonify({
                "success": False,
                "error": "Image validation failed",
                "error_code": "VALIDATION_FAILED",
                "issues": validation['issues']
            }), 400
        
        # Encode image for AI
        encoded_image = encode_image_for_api(image_file)
        if not encoded_image:
            return jsonify({
                "success": False,
                "error": "Failed to process image",
                "error_code": "ENCODING_FAILED"
            }), 500
        
        # Run AI detection
        detection_result = analyze_with_roboflow(encoded_image)
        
        if not detection_result['success']:
            return jsonify(detection_result), 500
        
        # Add image info to response
        detection_result.update({
            "image_info": {
                "width": validation['width'],
                "height": validation['height'],
                "file_size_mb": validation['file_size_mb'],
                "format": validation['format']
            },
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(detection_result)
        
    except Exception as e:
        logger.error(f"Error in detect_acne: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }), 500

@app.route('/detect-with-image', methods=['POST'])
def detect_with_annotated_image():
    """Detection endpoint that returns annotated image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400
        
        image_file = request.files['image']
        
        # Validate image
        validation = validate_image(image_file)
        if not validation['valid']:
            return jsonify({
                "success": False,
                "error": "Image validation failed",
                "issues": validation['issues']
            }), 400
        
        # Encode image for AI
        encoded_image = encode_image_for_api(image_file)
        if not encoded_image:
            return jsonify({
                "success": False,
                "error": "Failed to process image"
            }), 500
        
        # Run AI detection
        detection_result = analyze_with_roboflow(encoded_image)
        
        if not detection_result['success']:
            return jsonify(detection_result), 500
        
        # Create annotated image
        annotated_image = create_annotated_image(image_file, detection_result['predictions'])
        
        if annotated_image:
            return send_file(
                annotated_image,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f'mr_acne_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            )
        else:
            return jsonify({
                "success": False,
                "error": "Failed to create annotated image"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in detect_with_annotated_image: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/models', methods=['GET'])
def get_model_info():
    """Get information about available AI models"""
    return jsonify({
        "current_model": AI_MODEL,
        "capabilities": [
            "papules",
            "pustules", 
            "blackheads",
            "whiteheads",
            "dark spots"
        ],
        "requirements": {
            "min_resolution": "600x600",
            "recommended_resolution": "800x800+",
            "max_file_size": "10MB",
            "supported_formats": ["JPEG", "PNG"]
        }
    })

if __name__ == '__main__':
    # Railway sets PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

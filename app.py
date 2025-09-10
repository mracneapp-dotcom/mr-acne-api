#!/usr/bin/env python3
"""
Mr. Acne Detection API - COMPLETE WORKING VERSION WITH FULL USER JOURNEY
Flask web service for acne detection using Roboflow AI with complete frontend flow
"""

from flask import Flask, request, jsonify, send_file, Response
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

# Working AI model configuration - FIXED CONFIDENCE THRESHOLD
AI_MODEL = {
    "name": "facial-acne-detection-l06mq",
    "version": "1",
    "description": "Face-focused AI model",
    "confidence_threshold": 0.3  # LOWERED FROM 0.5 TO 0.3 - allows dark spots and other acne types to show
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
@app.route('/')
def home():
    """Serve the main application"""
    return FRONTEND_HTML

@app.route('/health')
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
    """JSON-only acne detection endpoint"""
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Validate image
        validation = validate_image(file)
        if not validation['valid']:
            return jsonify({
                "success": False,
                "error": "Image validation failed",
                "issues": validation['issues']
            }), 400

        # Encode image for API
        image_data = encode_image_for_api(file)
        if not image_data:
            return jsonify({"success": False, "error": "Failed to process image"}), 500

        # Analyze with Roboflow
        result = analyze_with_roboflow(image_data)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/detect-with-image', methods=['POST'])
def detect_with_image():
    """Endpoint that returns annotated image"""
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Validate image
        validation = validate_image(file)
        if not validation['valid']:
            return jsonify({
                "success": False,
                "error": "Image validation failed",
                "issues": validation['issues']
            }), 400

        # Create a copy for analysis
        file_copy = io.BytesIO(file.read())
        file.seek(0)

        # Encode image for API
        image_data = encode_image_for_api(file)
        if not image_data:
            return jsonify({"success": False, "error": "Failed to process image"}), 500

        # Analyze with Roboflow
        result = analyze_with_roboflow(image_data)
        
        if not result['success']:
            return jsonify(result), 500

        # Create annotated image if we have predictions
        if result.get('predictions') and len(result['predictions']) > 0:
            annotated_image = create_annotated_image(file, result['predictions'])
            if annotated_image:
                return send_file(
                    annotated_image,
                    mimetype='image/jpeg',
                    as_attachment=False,
                    download_name='annotated_results.jpg'
                )

        # If no predictions or annotation failed, return original image
        file.seek(0)
        return send_file(
            file,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='no_detections.jpg'
        )

    except Exception as e:
        logger.error(f"Detection with image error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500

# COMPLETE FRONTEND WITH ALL FEATURES - WORKING VERSION
FRONTEND_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mr. Acne - AI Acne Detection & Skincare Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #FDF5E6 0%, #7CB342 100%);
            min-height: 100vh;
            color: #000000;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        .section {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: none;
            animation: fadeIn 0.5s ease-in;
        }

        .section.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section-title {
            font-size: 1.8rem;
            color: #7CB342;
            margin-bottom: 20px;
            text-align: center;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .logo {
            font-size: 2.5rem;
            font-weight: bold;
            color: #7CB342;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 20px;
        }

        .btn {
            background: #7CB342;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
            font-weight: 600;
        }

        .btn:hover {
            background: #FF7A7A;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 122, 122, 0.3);
        }

        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-secondary {
            background: #FF7A7A;
        }

        .btn-secondary:hover {
            background: #7CB342;
            box-shadow: 0 5px 15px rgba(124, 179, 66, 0.3);
        }

        .btn-back {
            background: #ddd;
            color: #333;
        }

        .btn-back:hover {
            background: #bbb;
            transform: translateY(-1px);
        }

        .navigation-buttons {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
        }

        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #FDF5E6;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }

        .checkbox-item:hover {
            background: #7CB342;
            color: white;
            transform: translateY(-2px);
            border-color: #7CB342;
        }

        .checkbox-item input {
            margin-right: 12px;
            transform: scale(1.3);
        }

        .trial-info {
            background: linear-gradient(135deg, #fff3e0, #ffebee);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 2px solid #FF7A7A;
            text-align: center;
        }

        .uses-left {
            font-size: 1.2rem;
            color: #FF7A7A;
            font-weight: bold;
        }

        .upload-area {
            border: 3px dashed #7CB342;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #FDF5E6;
        }

        .upload-area:hover {
            border-color: #FF7A7A;
            background: #fff;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(124, 179, 66, 0.2);
        }

        .upload-area.dragover {
            border-color: #FF7A7A;
            background: #fff5f5;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 3rem;
            color: #7CB342;
            margin-bottom: 15px;
        }

        .file-input {
            display: none;
        }

        .browse-btn {
            background: #7CB342;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .browse-btn:hover {
            background: #FF7A7A;
            transform: translateY(-2px);
        }

        .image-preview {
            margin-top: 20px;
            text-align: center;
        }

        .preview-img {
            max-width: 300px;
            max-height: 300px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }

        .preview-img:hover {
            transform: scale(1.02);
        }

        .analyze-btn {
            background: #FF7A7A;
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 30px;
            font-size: 1.1rem;
            cursor: pointer;
            margin-top: 20px;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .analyze-btn:hover {
            background: #7CB342;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 122, 122, 0.3);
        }

        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .test-card {
            background: #FDF5E6;
            padding: 25px;
            margin: 15px 0;
            border-radius: 15px;
            border: 2px solid #7CB342;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .test-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(124, 179, 66, 0.3);
            border-color: #FF7A7A;
        }

        .test-results-form {
            background: #f0f8f0;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 1px solid #e0e0e0;
        }

        .radio-group {
            margin: 15px 0;
        }

        .radio-item {
            display: flex;
            align-items: center;
            padding: 12px;
            margin: 8px 0;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
        }

        .radio-item:hover {
            background: #e8f5e8;
            border-color: #7CB342;
            transform: translateX(5px);
        }

        .radio-item input {
            margin-right: 12px;
            transform: scale(1.2);
        }

        .routine-card {
            background: #FDF5E6;
            padding: 25px;
            margin: 15px 0;
            border-radius: 15px;
            border: 2px solid #7CB342;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .routine-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(124, 179, 66, 0.3);
            border-color: #FF7A7A;
        }

        .routine-title {
            font-size: 1.3rem;
            color: #7CB342;
            margin-bottom: 10px;
            font-weight: bold;
        }

        .ingredient-box {
            background: #e8f5e8;
            padding: 15px;
            margin: 12px 0;
            border-radius: 10px;
            border-left: 4px solid #7CB342;
        }

        .progress-bar {
            width: 100%;
            height: 12px;
            background: #e0e0e0;
            border-radius: 6px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #7CB342, #FF7A7A);
            border-radius: 6px;
            transition: width 0.5s ease;
            position: relative;
        }

        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shine 2s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .progress-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }

        .skin-profile {
            background: linear-gradient(135deg, #7CB342, #FF7A7A);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(124, 179, 66, 0.3);
        }

        .visual-results {
            margin: 20px 0;
            padding: 25px;
            background: #f0f8f0;
            border: 3px solid #7CB342;
            border-radius: 15px;
            text-align: center;
        }

        .visual-results h3 {
            color: #7CB342;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }

        .visual-results img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }

        .visual-results img:hover {
            transform: scale(1.02);
        }

        .email-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #7CB342;
            border-radius: 10px;
            font-family: Arial, sans-serif;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .email-input:focus {
            outline: none;
            border-color: #FF7A7A;
            box-shadow: 0 0 0 3px rgba(255, 122, 122, 0.1);
        }

        .loading {
            text-align: center;
            padding: 30px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #7CB342;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #c62828;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .info-box {
            background: #e3f2fd;
            padding: 18px;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
            line-height: 1.6;
        }

        .celebration {
            background: linear-gradient(135deg, #7CB342, #FF7A7A);
            color: white;
            padding: 30px;
            border-radius: 20px;
            margin: 25px 0;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .feature-highlight {
            background: linear-gradient(135deg, #e8f5e8, #fff3e0);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border-left: 5px solid #7CB342;
        }

        .step-explanation {
            background: rgba(124, 179, 66, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border: 1px solid rgba(124, 179, 66, 0.3);
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            animation: fadeIn 0.3s ease;
        }

        .modal-content {
            background-color: white;
            margin: 10% auto;
            padding: 30px;
            border-radius: 20px;
            width: 85%;
            max-width: 500px;
            text-align: center;
            animation: slideUp 0.3s ease;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }

        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }

            .checkbox-group {
                grid-template-columns: 1fr;
            }

            .section {
                padding: 20px;
            }

            .navigation-buttons {
                flex-direction: column;
                align-items: stretch;
            }

            .navigation-buttons .btn {
                margin: 5px 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">Mr. Acne</div>
            <div class="subtitle">AI-Powered Acne Detection & Skincare Assistant</div>
            <div class="trial-info">
                <div class="uses-left">Free Trial: <span id="usesLeft">3</span> analyses remaining</div>
                <small style="color: #666;">No signup required • Results in under 2 minutes</small>
            </div>
        </div>

        <!-- Progress Bar -->
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
        <div class="progress-labels">
            <span>0%</span>
            <span>100%</span>
        </div>

        <!-- Step 0: Welcome -->
        <div class="section active" id="step0">
            <div class="section-title">Welcome to Your AI Skincare Journey!</div>

            <div class="celebration">
                <h3 style="margin-bottom: 15px;">Ready to discover what your skin really needs?</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">Upload a photo and let our AI analyze your skin with medical-grade accuracy!</p>
            </div>

            <div class="feature-highlight">
                <h4 style="color: #7CB342; margin-bottom: 15px;">What makes this special:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: left;">
                    <div>
                        <strong>AI Photo Analysis</strong><br>
                        <small style="color: #666;">Advanced AI detects different types of acne with confidence scores</small>
                    </div>
                    <div>
                        <strong>Visual Results</strong><br>
                        <small style="color: #666;">See exactly where acne is detected with colored boxes and labels</small>
                    </div>
                    <div>
                        <strong>Detailed Analysis</strong><br>
                        <small style="color: #666;">Get papules, pustules, blackheads, whiteheads, and dark spots identified</small>
                    </div>
                    <div>
                        <strong>Instant Results</strong><br>
                        <small style="color: #666;">Complete analysis in under 30 seconds</small>
                    </div>
                </div>
            </div>

            <div class="info-box">
                <h4 style="color: #2196f3; margin-bottom: 15px;">How it works:</h4>
                <ol style="text-align: left; line-height: 1.8; margin-left: 20px;">
                    <li><strong>Upload your photo</strong> - Clear, well-lit face photo works best</li>
                    <li><strong>AI analysis</strong> - Our trained model examines your skin</li>
                    <li><strong>Visual results</strong> - See detections with colored boxes and confidence scores</li>
                    <li><strong>Complete assessment</strong> - Answer questions about your skin type and goals</li>
                    <li><strong>Get your routine</strong> - Receive personalized skincare recommendations</li>
                </ol>
                <p style="margin-top: 15px; font-weight: bold; color: #7CB342;">Total time: Under 5 minutes!</p>
            </div>

            <div style="text-align: center;">
                <button class="btn" style="font-size: 1.2rem; padding: 20px 45px; background: linear-gradient(135deg, #7CB342, #FF7A7A);" onclick="startJourney()">
                    Start My AI Analysis
                </button>
                <p style="margin-top: 15px; color: #666; font-size: 0.9rem;">
                    <em>100% free • No signup required • Medical-grade AI technology</em>
                </p>
            </div>
        </div>

        <!-- Step 1: Photo Upload -->
        <div class="section" id="step1">
            <div class="section-title">Upload Your Photo for AI Analysis</div>

            <div style="background: rgba(124, 179, 66, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid rgba(124, 179, 66, 0.3); text-align: center;">
                <p style="color: #7CB342; margin: 0; font-weight: 600;">
                    Our AI will analyze your photo to detect different types of acne with precise confidence scores and show you exactly where they are located.
                </p>
            </div>

            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div style="font-size: 1.2rem; color: #333; margin-bottom: 15px; font-weight: 600;">
                    Drag & drop your photo here or click to browse
                </div>
                <div style="color: #666; margin-bottom: 20px;">
                    Take a clear, well-lit selfie for best AI analysis results!
                </div>
                <div class="info-box">
                    <strong>For Best AI Analysis:</strong><br>
                    • Minimum size: 600x600 pixels for accurate detection<br>
                    • Clear face photo with good lighting (natural light works great!)<br>
                    • Face the camera directly, hair pulled back<br>
                    • Supported formats: JPG, PNG, WEBP (up to 10MB)
                </div>
                <button class="browse-btn" onclick="document.getElementById('fileInput').click()">
                    Choose Photo
                </button>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
            </div>

            <div class="image-preview" id="imagePreview"></div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <div>
                    <button class="analyze-btn" id="analyzeBtn" disabled>
                        Start AI Analysis
                    </button>
                </div>
            </div>
        </div>

        <!-- Step 2: AI Analysis Results -->
        <div class="section" id="step2">
            <div class="section-title">Your AI Analysis Results</div>
            <div id="analysisResults"></div>

            <div style="background: rgba(124, 179, 66, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid rgba(124, 179, 66, 0.3); text-align: center;">
                <p style="color: #7CB342; margin: 0; font-weight: 600;">
                    These results are generated by our trained AI model. Each detection shows the type of acne and the AI confidence level.
                </p>
            </div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Upload Another Photo</button>
                <div>
                    <button class="btn" onclick="nextStep(3)">Build My Complete Routine</button>
                    <button class="btn btn-secondary" onclick="showSkipWarning('assessment')">Skip Assessment</button>
                </div>
            </div>
        </div>

        <!-- Step 3: Skin Concerns -->
        <div class="section" id="step3">
            <div class="section-title">Tell Us About Your Skin Goals</div>

            <div class="step-explanation">
                <p style="color: #7CB342; margin: 0; text-align: center; font-weight: 600;">
                    <strong>Why this matters:</strong> Understanding all your skin concerns helps us create a comprehensive routine that addresses everything you want to improve, giving you better results faster!
                </p>
            </div>

            <p style="text-align: center; margin-bottom: 25px; font-size: 1.1rem; color: #333;">Besides acne, what would you like to improve? <strong>(Select all that apply)</strong></p>

            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="concern1" value="hyperpigmentation">
                    <label for="concern1">Dark spots/Hyperpigmentation</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern2" value="pores">
                    <label for="concern2">Large/Visible pores</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern3" value="wrinkles">
                    <label for="concern3">Fine lines/Wrinkles</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern4" value="dullness">
                    <label for="concern4">Dullness/Lack of glow</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern5" value="sensitivity">
                    <label for="concern5">Redness/Sensitivity</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern6" value="dryness">
                    <label for="concern6">Dryness/Flakiness</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern7" value="oiliness">
                    <label for="concern7">Excessive oiliness/Shine</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="concern8" value="texture">
                    <label for="concern8">Uneven texture/Bumpy skin</label>
                </div>
            </div>

            <div class="info-box">
                <strong>Pro Tip:</strong> Don't worry if you select multiple concerns! Modern skincare routines can address several issues at once with the right combination of ingredients.
            </div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <button class="btn" onclick="nextStep(4)">Continue to Skin Type Tests</button>
            </div>
        </div>

        <!-- Step 4: Skin Type Assessment -->
        <div class="section" id="step4">
            <div class="section-title">Discover Your Skin Type</div>

            <div class="step-explanation">
                <p style="color: #7CB342; margin: 0; text-align: center; font-weight: 600;">
                    <strong>Why skin type matters:</strong> Different skin types need different approaches. Oily skin needs oil control, dry skin needs hydration, combination skin needs balance. Knowing your type means better results!
                </p>
            </div>

            <div style="background: #fff3e0; padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
                <h4 style="color: #FF7A7A; margin-bottom: 15px;">Flexible Testing</h4>
                <p style="color: #666; line-height: 1.6;">
                    You can take <strong>1, 2, or all 3 tests</strong> - whatever fits your schedule! More tests = more accurate results, but even one test helps us personalize your routine.
                </p>
            </div>

            <div style="margin: 25px 0;">
                <div class="test-card" onclick="startTest('endday')">
                    <h4 style="color: #7CB342; margin-bottom: 10px;">Test 1: End-of-Day Check</h4>
                    <p style="margin-bottom: 10px;">Observe how your skin looks and feels right now at the end of your day.</p>
                    <div style="background: #e8f5e8; padding: 10px; border-radius: 8px;">
                        <small style="color: #388e3c; font-weight: 600;">Available immediately • 2 minutes</small>
                    </div>
                </div>

                <div class="test-card" onclick="startTest('blotting')">
                    <h4 style="color: #f57c00; margin-bottom: 10px;">Test 2: Blotting Paper Test</h4>
                    <p style="margin-bottom: 10px;">Press blotting papers on different face areas to measure oil production.</p>
                    <div class="info-box" style="margin: 15px 0 10px 0;">
                        <strong>What are blotting papers?</strong> Thin, absorbent papers that pick up oil from your skin. Available at any drugstore/pharmacy for $3-5, or use clean tissue as alternative!
                    </div>
                    <div style="background: #fff3e0; padding: 10px; border-radius: 8px;">
                        <small style="color: #f57c00; font-weight: 600;">Need blotting papers • 5 minutes</small>
                    </div>
                </div>

                <div class="test-card" onclick="startTest('overnight')">
                    <h4 style="color: #1976d2; margin-bottom: 10px;">Test 3: Overnight Assessment</h4>
                    <p style="margin-bottom: 10px;">Clean face before bed tonight, observe your skin tomorrow morning.</p>
                    <div style="background: #e3f2fd; padding: 10px; border-radius: 8px;">
                        <small style="color: #1976d2; font-weight: 600;">Best done tonight • Check tomorrow morning</small>
                    </div>
                </div>
            </div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <div>
                    <button class="btn btn-secondary" onclick="setReminder()">Remind Me Tonight</button>
                    <button class="btn" onclick="showSkipWarning('tests')">Skip Tests (General Results)</button>
                </div>
            </div>
        </div>

        <!-- Step 5: Test Execution -->
        <div class="section" id="step5">
            <div class="section-title">Complete Your Skin Test</div>
            <div id="testContent"></div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <div id="testNavigation"></div>
            </div>
        </div>

        <!-- Step 6: Skin Profile Results -->
        <div class="section" id="step6">
            <div class="section-title">Amazing! Your Personal Skin Profile Is Ready!</div>

            <div class="celebration">
                <h3 style="margin-bottom: 15px;">Mission Accomplished!</h3>
                <p style="font-size: 1.1rem;">We've analyzed your skin and created your unique profile. Here's what we discovered...</p>
            </div>

            <div class="skin-profile" id="skinProfile">
                <h3 style="margin-bottom: 15px; text-align: center;">Your Personal Skin Analysis:</h3>
                <div id="profileContent"></div>
            </div>

            <div id="skinTypeExplanation" style="background: #f8f9fa; padding: 25px; border-radius: 15px; margin: 20px 0;"></div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <button class="btn" onclick="nextStep(7)" style="font-size: 1.2rem; padding: 18px 35px; background: linear-gradient(135deg, #7CB342, #FF7A7A);">
                    Show Me My Routine Options!
                </button>
            </div>
        </div>

        <!-- Step 7: Routine Selection -->
        <div class="section" id="step7">
            <div class="section-title">Choose Your Perfect Routine Level</div>

            <div class="step-explanation">
                <p style="color: #7CB342; margin: 0; text-align: center; font-weight: 600;">
                    <strong>Think of this like choosing your workout intensity!</strong> We recommend starting with Foundation routine - it's like building a strong base. You can always level up later as your skin adapts.
                </p>
            </div>

            <div class="routine-card" onclick="selectRoutine('foundation')" style="border: 3px solid #7CB342;">
                <div class="routine-title">Foundation Routine (Recommended)</div>
                <div style="color: #666; margin-bottom: 15px; font-weight: 600;">3 essential steps: Cleanse → Treat → Protect</div>
                <p style="line-height: 1.6;">Perfect starting point with proven basics. Gentle yet effective - ideal for beginners or anyone wanting a simple, reliable routine that actually works!</p>
                <div style="background: #e8f5e8; padding: 10px; border-radius: 8px; margin-top: 15px;">
                    <small style="color: #388e3c; font-weight: 600;">Most popular choice • Great for sensitive skin</small>
                </div>
            </div>

            <div class="routine-card" onclick="selectRoutine('balanced')">
                <div class="routine-title">Balanced Routine</div>
                <div style="color: #666; margin-bottom: 15px; font-weight: 600;">5-6 steps: Enhanced care with targeted treatments</div>
                <p style="line-height: 1.6;">Add serums and targeted treatments for specific concerns. Perfect for those ready to address multiple skin goals with a comprehensive approach.</p>
                <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin-top: 15px;">
                    <small style="color: #f57c00; font-weight: 600;">Coming soon • Foundation available now</small>
                </div>
            </div>

            <div class="routine-card" onclick="selectRoutine('intensive')">
                <div class="routine-title">Intensive Routine</div>
                <div style="color: #666; margin-bottom: 15px; font-weight: 600;">8+ steps: Complete professional-level regimen</div>
                <p style="line-height: 1.6;">Advanced multi-step routine for experienced skincare enthusiasts who want maximum results and enjoy detailed self-care rituals.</p>
                <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin-top: 15px;">
                    <small style="color: #f57c00; font-weight: 600;">Coming soon • Foundation available now</small>
                </div>
            </div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <div></div>
            </div>
        </div>

        <!-- Step 8: Routine Display -->
        <div class="section" id="step8">
            <div class="section-title">Your Personalized Skincare Routine</div>

            <div id="routineContent"></div>

            <div style="background: linear-gradient(135deg, #fff3e0, #ffebee); padding: 25px; border-radius: 15px; margin: 25px 0; border: 2px solid #FF7A7A;">
                <h4 style="color: #FF7A7A; margin-bottom: 15px; text-align: center;">Save Your Routine & Stay Connected</h4>
                <p style="margin-bottom: 20px; text-align: center; color: #666;">Want to keep this routine handy and get skincare tips? We'll email your personalized routine plus bonus tips!</p>

                <input type="email" class="email-input" placeholder="Enter your email for your routine + bonus tips" id="emailInput" style="margin-bottom: 15px;">

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <button class="btn" onclick="saveRoutineToEmail()" style="background: #FF7A7A;">
                        Email My Routine
                    </button>
                    <button class="btn btn-secondary" onclick="saveToProfile()">
                        Save to Profile
                    </button>
                    <button class="btn" onclick="shareRoutine()" style="background: #7CB342;">
                        Share with Friends
                    </button>
                </div>

                <p style="margin-top: 15px; text-align: center; font-size: 0.9rem; color: #666;">
                    <em>We respect your privacy • No spam, just helpful skincare tips • Unsubscribe anytime</em>
                </p>
            </div>

            <div class="navigation-buttons">
                <button class="btn btn-back" onclick="previousStep()">← Back</button>
                <div>
                    <button class="btn" onclick="restartAssessment()">Start New Assessment</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Skip Warning Modal -->
    <div id="skipModal" class="modal">
        <div class="modal-content">
            <div style="font-size: 3rem; margin-bottom: 20px;">🤔</div>
            <h3 style="color: #FF7A7A; margin-bottom: 20px;">Hold on a second!</h3>
            <p style="margin-bottom: 20px; line-height: 1.6;">
                When you skip the assessment, we can't build your routine as accurately as we'd like.
                The more we know about your skin, the better results we can give you!
            </p>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <p style="margin: 0; color: #7CB342; font-weight: bold;">
                    We promise it's going to be quick, fun, and totally worth it!
                </p>
            </div>
            <div style="margin-top: 25px;">
                <button class="btn" onclick="closeSkipModal()" style="padding: 15px 30px;">
                    Let's Do This Right!
                </button>
                <button class="btn btn-secondary" onclick="confirmSkip()">
                    Skip Anyway
                </button>
            </div>
        </div>
    </div>

    <script>
        // Application State
        var currentStep = 0;
        var totalSteps = 8;
        var stepHistory = [0];
        var selectedFile = null;
        var usesLeft = parseInt(localStorage.getItem('mrAcneUses') || '3');
        var analysisResult = null;
        var skinProfile = {};
        var selectedConcerns = [];
        var completedTests = [];
        var testResults = {};
        var currentTestType = null;
        var skipTarget = '';
        var userEmail = '';

        // CRITICAL: Start journey function - MUST WORK
        function startJourney() {
            console.log('Starting AI analysis journey...');
            
            try {
                var step0 = document.getElementById('step0');
                var step1 = document.getElementById('step1');

                if (step0 && step1) {
                    step0.classList.remove('active');
                    step1.classList.add('active');
                    currentStep = 1;
                    stepHistory.push(0);
                    updateProgress();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    console.log('Successfully moved to step 1');
                } else {
                    console.error('Could not find step elements');
                    alert('Navigation error. Please refresh the page.');
                }
            } catch (error) {
                console.error('Error in startJourney:', error);
                alert('Navigation error. Please refresh the page and try again.');
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            updateUsesDisplay();
            initEventListeners();
            updateProgress();
            console.log('Mr. Acne AI Analysis initialized successfully');
        });

        function updateProgress() {
            var percentage = Math.round((currentStep / totalSteps) * 100);
            var progressFill = document.getElementById('progressFill');
            if (progressFill) {
                progressFill.style.width = percentage + '%';
            }
        }

        function nextStep(step) {
            console.log('Moving to step:', step);
            if (stepHistory[stepHistory.length - 1] !== currentStep) {
                stepHistory.push(currentStep);
            }

            var currentSection = document.getElementById('step' + currentStep);
            var targetSection = document.getElementById('step' + step);

            if (currentSection && targetSection) {
                currentSection.classList.remove('active');
                targetSection.classList.add('active');
                currentStep = step;
                updateProgress();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }

        function previousStep() {
            if (stepHistory.length > 1) {
                stepHistory.pop();
                var previousStepNum = stepHistory[stepHistory.length - 1];
                nextStep(previousStepNum);
            }
        }

        function updateUsesDisplay() {
            var usesElement = document.getElementById('usesLeft');
            if (usesElement) {
                usesElement.textContent = usesLeft;
                if (usesLeft <= 1) {
                    usesElement.style.color = '#FF7A7A';
                    usesElement.style.fontWeight = 'bold';
                }
            }
        }

        function initEventListeners() {
            var uploadArea = document.getElementById('uploadArea');
            var fileInput = document.getElementById('fileInput');
            var analyzeBtn = document.getElementById('analyzeBtn');

            if (uploadArea && fileInput) {
                uploadArea.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });

                uploadArea.addEventListener('dragleave', function() {
                    uploadArea.classList.remove('dragover');
                });

                uploadArea.addEventListener('drop', function(e) {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    var files = e.dataTransfer.files;
                    if (files.length > 0) {
                        handleFileSelect(files[0]);
                    }
                });

                fileInput.addEventListener('change', function(e) {
                    if (e.target.files.length > 0) {
                        handleFileSelect(e.target.files[0]);
                    }
                });

                uploadArea.addEventListener('click', function() {
                    fileInput.click();
                });
            }

            if (analyzeBtn) {
                analyzeBtn.addEventListener('click', function() {
                    analyzeImage();
                });
            }
        }

        function handleFileSelect(file) {
            if (!validateFile(file)) {
                return;
            }

            selectedFile = file;
            showImagePreview(file);
            var analyzeBtn = document.getElementById('analyzeBtn');
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }

            var uploadArea = document.getElementById('uploadArea');
            uploadArea.style.borderColor = '#7CB342';
            uploadArea.style.background = '#e8f5e8';
        }

        function validateFile(file) {
            var validTypes = ['image/jpeg', 'image/png', 'image/webp'];
            if (validTypes.indexOf(file.type) === -1) {
                showError('Please select a valid image file (JPG, PNG, or WEBP)');
                return false;
            }

            if (file.size > 10 * 1024 * 1024) {
                showError('File size must be less than 10MB');
                return false;
            }

            return true;
        }

        function showImagePreview(file) {
            var reader = new FileReader();
            reader.onload = function(e) {
                var img = new Image();
                img.onload = function() {
                    if (img.width < 600 || img.height < 600) {
                        showError('Image is too small (' + img.width + 'x' + img.height + '). Minimum size is 600x600 pixels for accurate AI analysis.');
                        return;
                    }

                    var preview = document.getElementById('imagePreview');
                    preview.innerHTML = '<div style="animation: fadeIn 0.5s ease;"><img src="' + e.target.result + '" alt="Preview" class="preview-img"><div style="margin-top: 15px; color: #7CB342; font-weight: 600;">Perfect! Image size: ' + img.width + 'x' + img.height + ' pixels</div><div style="margin-top: 10px; color: #666;">Ready for AI analysis</div></div>';
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function analyzeImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            if (usesLeft <= 0) {
                alert('You have used all your free analyses! More features coming soon.');
                return;
            }

            showLoading();

            try {
                var formData = new FormData();
                formData.append('image', selectedFile);

                // FIRST: Get JSON analysis results
                fetch('/detect', {
                    method: 'POST',
                    body: formData
                })
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error('Detection failed: ' + response.status);
                    }
                    return response.json();
                })
                .then(function(result) {
                    analysisResult = result;

                    // SECOND: Get annotated image if we have detections
                    if (result.success && result.predictions && result.predictions.length > 0) {
                        var imageFormData = new FormData();
                        imageFormData.append('image', selectedFile);

                        fetch('/detect-with-image', {
                            method: 'POST',
                            body: imageFormData
                        })
                        .then(function(imageResponse) {
                            if (imageResponse.ok) {
                                return imageResponse.blob();
                            }
                            throw new Error('Visual analysis failed');
                        })
                        .then(function(imageBlob) {
                            var imageUrl = URL.createObjectURL(imageBlob);
                            displayAnalysisResults(result, imageUrl);
                        })
                        .catch(function(imageError) {
                            console.error('Error getting annotated image:', imageError);
                            displayAnalysisResults(result, null);
                        });
                    } else {
                        displayAnalysisResults(result, null);
                    }

                    decrementUses();
                    nextStep(2);
                })
                .catch(function(error) {
                    console.error('Analysis error:', error);
                    showError('AI analysis failed: ' + error.message + '. Please try again.');
                });

            } catch (error) {
                console.error('Analysis error:', error);
                showError('AI analysis failed. Please try again.');
            }
        }

        function displayAnalysisResults(result, imageUrl) {
            var resultsDiv = document.getElementById('analysisResults');

            if (!result.success) {
                resultsDiv.innerHTML = '<div class="error"><h3>Analysis Failed</h3><p>' + (result.error || 'Unknown error occurred') + '</p></div>';
                return;
            }

            console.log('Displaying results. Image URL:', imageUrl);
            console.log('Predictions:', result.predictions);

            // Build the results HTML - START WITH VISUAL RESULTS
            var html = '<div style="text-align: center; margin-bottom: 25px;"><div style="font-size: 1.4rem; color: #FF7A7A; font-weight: bold; margin-bottom: 10px;">AI Analysis Complete: ' + (result.total_found || 0) + ' detection' + (result.total_found !== 1 ? 's' : '') + ' found</div><p style="color: #666; font-size: 1.1rem; line-height: 1.6;">Our AI has analyzed your photo and identified the following:</p></div>';

            // ALWAYS show visual results section - MOST IMPORTANT
            if (imageUrl) {
                console.log('Adding visual results with image');
                html += '<div class="visual-results" style="margin: 30px 0; padding: 30px; background: #f0f8f0; border: 3px solid #7CB342; border-radius: 15px; text-align: center;"><h3 style="color: #7CB342; margin-bottom: 20px; font-size: 1.8rem;">🎯 Visual Detection Results</h3><img src="' + imageUrl + '" alt="AI Acne Detection Results" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);"><p style="color: #666; margin-top: 20px; font-style: italic; line-height: 1.6; font-size: 1.1rem;"><strong>Your photo with AI-detected acne spots marked with colored boxes and confidence scores.</strong><br>Each colored box represents a different detection with its accuracy level.</p></div>';
            } else {
                console.log('No image URL provided, showing message');
                html += '<div style="background: #fff3e0; padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center; border: 2px solid #f57c00;"><h4 style="color: #f57c00; margin-bottom: 10px;">Visual Results Processing...</h4><p style="color: #666;">We are working on generating your visual results. The analysis data is shown below.</p></div>';
            }

            if (!result.predictions || result.predictions.length === 0) {
                html += '<div class="celebration"><h3 style="margin-bottom: 15px;">Great News!</h3><p style="font-size: 1.1rem; line-height: 1.6;">Our AI analysis did not detect any visible acne in your photo! Your skin looks healthy and clear.</p></div><div style="background: #e8f5e8; padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;"><h4 style="color: #7CB342; margin-bottom: 10px;">What This Means</h4><p style="color: #666; line-height: 1.6;">Our trained AI model examined your photo and did not identify any acne lesions, which is excellent news for your skin health!</p></div>';
            } else {
                // Add detailed findings
                html += '<div style="margin-top: 25px;"><h4 style="color: #7CB342; margin-bottom: 20px; text-align: center; font-size: 1.5rem;">📊 Detailed AI Findings:</h4>';

                // Group predictions by type for better presentation
                var groupedPredictions = {};
                result.predictions.forEach(function(prediction, index) {
                    var type = prediction.class;
                    if (!groupedPredictions[type]) {
                        groupedPredictions[type] = [];
                    }
                    groupedPredictions[type].push({
                        confidence: prediction.confidence,
                        x: prediction.x,
                        y: prediction.y,
                        index: index + 1
                    });
                });

                // Display each acne type found
                Object.keys(groupedPredictions).forEach(function(type) {
                    var predictions = groupedPredictions[type];
                    var typeName = type.charAt(0).toUpperCase() + type.slice(1);

                    // Get emoji for acne type
                    var typeEmoji = '';
                    if (type.toLowerCase() === 'papules') typeEmoji = '🔴';
                    else if (type.toLowerCase() === 'pustules') typeEmoji = '🟡';
                    else if (type.toLowerCase() === 'blackheads') typeEmoji = '⚫';
                    else if (type.toLowerCase() === 'whiteheads') typeEmoji = '⚪';
                    else if (type.toLowerCase() === 'dark_spots' || type.toLowerCase() === 'dark spot') typeEmoji = '🔵';
                    else typeEmoji = '🎯';

                    html += '<div style="background: #FDF5E6; padding: 20px; margin: 15px 0; border-radius: 12px; border-left: 5px solid #7CB342;"><h4 style="color: #7CB342; margin-bottom: 15px;">' + typeEmoji + ' ' + typeName + ' (' + predictions.length + ' detected)</h4><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">';

                    predictions.forEach(function(prediction) {
                        var confidence = Math.round(prediction.confidence * 100);
                        var confidenceColor = confidence >= 70 ? '#4CAF50' : confidence >= 50 ? '#FF9800' : '#FF5722';

                        html += '<div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e0e0e0;"><strong>Detection #' + prediction.index + '</strong><br><span style="color: ' + confidenceColor + '; font-weight: bold;">Confidence: ' + confidence + '%</span><br><small style="color: #666;">Location: (' + Math.round(prediction.x) + ', ' + Math.round(prediction.y) + ')</small></div>';
                    });

                    html += '</div></div>';
                });

                html += '</div>';
            }

            // Add AI model information
            html += '<div style="background: #e3f2fd; padding: 20px; border-radius: 15px; margin: 20px 0; border-left: 4px solid #2196f3;"><h4 style="color: #2196f3; margin-bottom: 10px;">🤖 AI Model Information</h4><p style="color: #666; line-height: 1.6; margin: 0;">Analysis performed by: <strong>' + result.model_used + '</strong><br>Processing time: ' + (result.processing_time || 'N/A') + 'ms<br>Detection technology: Medical-grade AI trained on thousands of acne images</p></div>';

            resultsDiv.innerHTML = html;
            console.log('Results displayed successfully');
        }

        function showLoading() {
            var resultsDiv = document.getElementById('analysisResults');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 10px;">AI is analyzing your skin...</div><div style="color: #666;">Our trained model is examining your photo for different types of acne</div><div style="margin-top: 15px; color: #7CB342; font-size: 0.9rem;">This usually takes 5-10 seconds</div></div>';

            nextStep(2);
            var step2Element = document.getElementById('step2');
            if (step2Element) {
                step2Element.scrollIntoView({ behavior: 'smooth' });
            }
        }

        function decrementUses() {
            usesLeft--;
            localStorage.setItem('mrAcneUses', usesLeft.toString());
            updateUsesDisplay();
        }

        function showError(message) {
            var existingError = document.querySelector('.error');
            if (existingError) {
                existingError.remove();
            }

            var error = document.createElement('div');
            error.className = 'error';
            error.innerHTML = '<strong>Error:</strong> ' + message + '<button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: #c62828; font-size: 1.2rem; cursor: pointer;">×</button>';

            var activeSection = document.querySelector('.section.active');
            if (activeSection) {
                activeSection.appendChild(error);

                setTimeout(function() {
                    if (error.parentElement) {
                        error.remove();
                    }
                }, 7000);
            }
        }

        // NEW FUNCTIONS FOR EXTENDED USER JOURNEY
        function showSkipWarning(target) {
            skipTarget = target;
            document.getElementById('skipModal').style.display = 'block';
        }

        function closeSkipModal() {
            document.getElementById('skipModal').style.display = 'none';
        }

        function confirmSkip() {
            closeSkipModal();
            if (skipTarget === 'assessment') {
                skipToRoutines();
            } else if (skipTarget === 'tests') {
                skipTests();
            }
        }

        function skipToRoutines() {
            skinProfile = {
                acne: analysisResult ? analysisResult.total_found : 0,
                skinType: 'unknown',
                concerns: [],
                hasTestResults: false,
                skippedAssessment: true
            };
            nextStep(7);
        }

        function skipTests() {
            collectConcerns();
            skinProfile = {
                acne: analysisResult ? analysisResult.total_found : 0,
                skinType: 'normal',
                concerns: selectedConcerns,
                hasTestResults: false,
                skippedTests: true
            };
            displaySkinProfile();
            nextStep(6);
        }

        function collectConcerns() {
            selectedConcerns = [];
            var checkboxes = document.querySelectorAll('#step3 input[type="checkbox"]:checked');
            checkboxes.forEach(function(cb) {
                selectedConcerns.push(cb.value);
            });
        }

        function startTest(testType) {
            collectConcerns();
            currentTestType = testType;

            var testNames = {
                'overnight': 'Overnight Clean Face Test',
                'endday': 'End-of-Day Assessment',
                'blotting': 'Blotting Paper Test'
            };

            var testInstructions = {
                'overnight': '<h4 style="color: #1976d2; margin-bottom: 15px;">Tonight Instructions:</h4><ol style="line-height: 1.8; padding-left: 20px; font-size: 1.1rem;"><li>Wash your face thoroughly with your usual cleanser</li><li>Do not apply any products (no moisturizer, serums, treatments, etc.)</li><li>Go to sleep as normal</li><li>In the morning, examine your skin before washing or applying anything</li></ol><div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 15px 0;"><strong>What we are testing:</strong> Your skin natural oil production overnight helps us determine if you have oily, dry, normal, or combination skin.</div>',
                'endday': '<h4 style="color: #7CB342; margin-bottom: 15px;">Right Now Instructions:</h4><ol style="line-height: 1.8; padding-left: 20px; font-size: 1.1rem;"><li>Look at your face in good lighting (natural light is best)</li><li>Observe how your skin feels and looks after a full day</li><li>Pay attention to oily areas, dry patches, or shine</li><li>Note any tightness, comfort, or other sensations</li></ol><div style="background: #e8f5e8; padding: 15px; border-radius: 10px; margin: 15px 0;"><strong>What we are testing:</strong> How your skin behaves throughout a normal day tells us about your skin oil production patterns and needs.</div>',
                'blotting': '<h4 style="color: #f57c00; margin-bottom: 15px;">Blotting Test Instructions:</h4><ol style="line-height: 1.8; padding-left: 20px; font-size: 1.1rem;"><li>Wash your face and wait 1 hour without applying products</li><li>Press blotting papers gently on: forehead, nose, cheeks, chin</li><li>Hold for 10-15 seconds on each area with gentle pressure</li><li>Examine the papers for oil spots or transparency</li></ol><div style="background: #fff3e0; padding: 15px; border-radius: 10px; margin: 15px 0;"><strong>What we are testing:</strong> Direct measurement of oil production in different face zones gives us the most accurate skin type assessment.</div>'
            };

            document.getElementById('testContent').innerHTML = '<div style="text-align: center; margin-bottom: 25px;"><h3 style="color: #7CB342;">' + testNames[testType] + '</h3><p style="color: #666; font-size: 1.1rem;">This test helps us understand your skin unique characteristics</p></div><div style="background: #f0f8f0; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 2px solid #7CB342;">' + testInstructions[testType] + '</div><div style="text-align: center; margin-top: 25px;"><button class="btn" onclick="showTestResultsForm()" style="padding: 15px 35px; font-size: 1.1rem;">I have Completed The Test</button><button class="btn btn-secondary" onclick="saveTestForLater()" style="padding: 15px 35px;">I will Do This Later</button></div>';
            nextStep(5);
        }

        function showTestResultsForm() {
            var testForms = {
                'overnight': '<h4 style="color: #1976d2; margin-bottom: 20px; text-align: center;">What did you observe this morning?</h4><div class="radio-group"><div class="radio-item"><input type="radio" name="overnightResult" value="oily" id="overnight_oily"><label for="overnight_oily"><strong>Very oily/shiny all over face</strong><br><small style="color: #666;">Face feels greasy, visible shine everywhere</small></label></div><div class="radio-item"><input type="radio" name="overnightResult" value="combination" id="overnight_combo"><label for="overnight_combo"><strong>Oily T-zone, normal/dry cheeks</strong><br><small style="color: #666;">Shine on forehead, nose, chin but cheeks feel normal or dry</small></label></div><div class="radio-item"><input type="radio" name="overnightResult" value="normal" id="overnight_normal"><label for="overnight_normal"><strong>Comfortable, balanced</strong><br><small style="color: #666;">Not oily or dry, skin feels comfortable</small></label></div><div class="radio-item"><input type="radio" name="overnightResult" value="dry" id="overnight_dry"><label for="overnight_dry"><strong>Tight, dry, or flaky</strong><br><small style="color: #666;">Skin feels tight, may see dry patches or flaking</small></label></div></div>',
                'endday': '<h4 style="color: #7CB342; margin-bottom: 20px; text-align: center;">How does your skin look and feel right now?</h4><div class="radio-group"><div class="radio-item"><input type="radio" name="enddayResult" value="oily" id="endday_oily"><label for="endday_oily"><strong>Shiny, oily, or greasy feeling</strong><br><small style="color: #666;">Visible shine, feels slippery or greasy to touch</small></label></div><div class="radio-item"><input type="radio" name="enddayResult" value="combination" id="endday_combo"><label for="endday_combo"><strong>Mixed - oily in some areas, normal/dry in others</strong><br><small style="color: #666;">T-zone (forehead, nose, chin) oily but cheeks different</small></label></div><div class="radio-item"><input type="radio" name="enddayResult" value="normal" id="endday_normal"><label for="endday_normal"><strong>Balanced and comfortable</strong><br><small style="color: #666;">No excessive oil or dryness, feels pleasant</small></label></div><div class="radio-item"><input type="radio" name="enddayResult" value="dry" id="endday_dry"><label for="endday_dry"><strong>Dry, tight, or needs moisturizer</strong><br><small style="color: #666;">Feels tight, rough, or like you need moisturizer</small></label></div></div>',
                'blotting': '<h4 style="color: #f57c00; margin-bottom: 20px; text-align: center;">What did the blotting papers show?</h4><div class="radio-group"><div class="radio-item"><input type="radio" name="blottingResult" value="oily" id="blotting_oily"><label for="blotting_oily"><strong>Heavy oil on all papers</strong><br><small style="color: #666;">Papers from all areas show significant oil/transparency</small></label></div><div class="radio-item"><input type="radio" name="blottingResult" value="combination" id="blotting_combo"><label for="blotting_combo"><strong>Oil on T-zone papers only</strong><br><small style="color: #666;">Forehead, nose, chin papers show oil; cheek papers clean</small></label></div><div class="radio-item"><input type="radio" name="blottingResult" value="normal" id="blotting_normal"><label for="blotting_normal"><strong>Light oil on some papers</strong><br><small style="color: #666;">Minimal oil detected, papers mostly clean</small></label></div><div class="radio-item"><input type="radio" name="blottingResult" value="dry" id="blotting_dry"><label for="blotting_dry"><strong>Little to no oil on any papers</strong><br><small style="color: #666;">All papers remain clean and dry</small></label></div></div>'
            };

            document.getElementById('testContent').innerHTML = '<div style="text-align: center; margin-bottom: 25px;"><h3 style="color: #7CB342;">Record Your Test Results</h3><p style="color: #666;">Choose the option that best describes what you observed</p></div><div class="test-results-form">' + testForms[currentTestType] + '</div><div style="text-align: center; margin-top: 25px;"><button class="btn" onclick="saveTestResults()" style="padding: 15px 35px; font-size: 1.1rem;">Save Results & Continue</button></div>';
        }

        function saveTestResults() {
            var resultInputs = document.querySelectorAll('input[type="radio"]:checked');
            if (resultInputs.length === 0) {
                showError('Please select a result before continuing');
                return;
            }
            var result = resultInputs[0].value;
            testResults[currentTestType] = result;
            completedTests.push(currentTestType);
            var skinType = determineSkinType();
            skinProfile = {
                acne: analysisResult ? analysisResult.total_found : 0,
                skinType: skinType,
                concerns: selectedConcerns,
                testsCompleted: completedTests.length,
                testResults: testResults,
                hasTestResults: true
            };
            displaySkinProfile();
            nextStep(6);
        }

        function saveTestForLater() {
            showError('Feature coming soon! For now, you can skip tests or take them immediately.');
        }

        function setReminder() {
            showError('Reminder feature coming soon! Please bookmark this page and return later.');
        }

        function determineSkinType() {
            if (Object.keys(testResults).length === 0) {
                return 'unknown';
            }
            var results = Object.values(testResults);
            var counts = {
                'oily': results.filter(function(r) { return r === 'oily'; }).length,
                'dry': results.filter(function(r) { return r === 'dry'; }).length,
                'combination': results.filter(function(r) { return r === 'combination'; }).length,
                'normal': results.filter(function(r) { return r === 'normal'; }).length
            };
            var maxCount = Math.max.apply(Math, Object.values(counts));
            var predominantTypes = Object.keys(counts).filter(function(key) { return counts[key] === maxCount; });
            if (predominantTypes.indexOf('combination') !== -1) {
                return 'combination';
            } else if (predominantTypes.indexOf('oily') !== -1) {
                return 'oily';
            } else if (predominantTypes.indexOf('dry') !== -1) {
                return 'dry';
            } else {
                return 'normal';
            }
        }

        function displaySkinProfile() {
            var skinType = skinProfile.skinType;
            var acne = skinProfile.acne;
            var concerns = skinProfile.concerns;
            var hasTestResults = skinProfile.hasTestResults;

            document.getElementById('profileContent').innerHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;"><div><div style="font-size: 2rem; margin-bottom: 10px;">🎯</div><strong>Acne Spots Detected</strong><br><span style="font-size: 1.5rem; color: #FFD700;">' + acne + '</span></div><div><div style="font-size: 2rem; margin-bottom: 10px;">🧪</div><strong>Skin Type</strong><br><span style="font-size: 1.2rem; color: #FFD700;">' + skinType.charAt(0).toUpperCase() + skinType.slice(1) + '</span></div><div><div style="font-size: 2rem; margin-bottom: 10px;">🎯</div><strong>Additional Goals</strong><br><span style="color: #FFD700;">' + concerns.length + ' selected</span></div><div><div style="font-size: 2rem; margin-bottom: 10px;">📊</div><strong>Tests Completed</strong><br><span style="color: #FFD700;">' + completedTests.length + '</span></div></div>' + (hasTestResults ? '<div style="text-align: center; margin-top: 20px; color: #FFD700;">Results based on actual skin assessment</div>' : '');

            var explanation = '';
            if (hasTestResults) {
                var skinTypeExplanations = {
                    'oily': {
                        description: 'Your skin produces more oil than average throughout the day.',
                        approach: 'We will focus on gentle oil control, lightweight products, and non-comedogenic formulas that will not clog your pores.',
                        benefits: 'The plus side? Oily skin often ages more slowly and stays hydrated naturally!'
                    },
                    'dry': {
                        description: 'Your skin needs extra hydration and barrier support.',
                        approach: 'We will recommend richer moisturizers, hydrating ingredients, and gentle products that support your skin barrier.',
                        benefits: 'With the right routine, dry skin can become beautifully smooth and comfortable!'
                    },
                    'combination': {
                        description: 'You have different needs in different areas - usually oily T-zone and normal/dry cheeks.',
                        approach: 'We will create a balanced routine that addresses both oily and dry areas without over-treating either.',
                        benefits: 'With the right approach, combination skin can achieve perfect balance!'
                    },
                    'normal': {
                        description: 'Lucky you! Your skin is well-balanced with minimal concerns.',
                        approach: 'We will focus on maintaining this balance while addressing your specific acne concerns.',
                        benefits: 'Normal skin responds beautifully to consistent, gentle care!'
                    },
                    'unknown': {
                        description: 'Since we do not have test results, we will use a gentle approach suitable for most skin types.',
                        approach: 'We will create a balanced routine that works well for most people while being gentle enough for sensitive skin.',
                        benefits: 'You can always come back and take the tests later for more personalized results!'
                    }
                };

                var skinInfo = skinTypeExplanations[skinType];
                explanation = '<h4 style="color: #333; margin-bottom: 15px; text-align: center;">Understanding Your ' + skinType.charAt(0).toUpperCase() + skinType.slice(1) + ' Skin</h4><div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 15px;"><p style="margin-bottom: 15px; line-height: 1.6; font-size: 1.1rem;"><strong>What this means:</strong> ' + skinInfo.description + '</p><p style="margin-bottom: 15px; line-height: 1.6;"><strong>Our approach:</strong> ' + skinInfo.approach + '</p><p style="line-height: 1.6; color: #7CB342; font-weight: 600;"><strong>The good news:</strong> ' + skinInfo.benefits + '</p></div>';
            }

            document.getElementById('skinTypeExplanation').innerHTML = explanation;
        }

        function selectRoutine(routineType) {
            if (routineType === 'foundation') {
                generateFoundationRoutine();
                nextStep(8);
            } else {
                showError('Coming soon! Foundation routine is available now.');
            }
        }

        function generateFoundationRoutine() {
            var skinType = skinProfile.skinType || 'normal';
            var acneCount = skinProfile.acne || 0;
            var concerns = skinProfile.concerns || [];

            var routineHTML = '<div class="celebration"><h3 style="margin-bottom: 15px;">Your Foundation Routine is Ready!</h3><p style="font-size: 1.1rem;">A simple yet effective 3-step routine personalized for your ' + skinType + ' skin with ' + acneCount + ' acne detections.</p></div>';

            // Morning routine
            routineHTML += '<div style="background: #fff3e0; padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 5px solid #f57c00;"><h4 style="color: #f57c00; margin-bottom: 20px;">Morning Routine (5 minutes)</h4>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 1: Gentle Cleanser</h5><p style="margin-bottom: 10px;"><strong>For ' + skinType + ' skin:</strong> Use a gentle, non-stripping cleanser with salicylic acid for acne prevention.</p><p style="color: #666; font-size: 0.9rem;"><em>Look for: Salicylic acid (0.5-2%), gentle surfactants, pH-balanced formula</em></p></div>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 2: Treatment</h5><p style="margin-bottom: 10px;"><strong>Acne spots detected:</strong> Apply a gentle acne treatment with benzoyl peroxide or niacinamide.</p><p style="color: #666; font-size: 0.9rem;"><em>Look for: Benzoyl peroxide (2.5%), Niacinamide (5-10%), or gentle retinoids</em></p></div>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 3: Protect</h5><p style="margin-bottom: 10px;"><strong>SPF is crucial:</strong> Use broad-spectrum SPF 30+ that will not clog pores.</p><p style="color: #666; font-size: 0.9rem;"><em>Look for: Zinc oxide, titanium dioxide, or chemical SPF labeled "non-comedogenic"</em></p></div>';

            routineHTML += '</div>';

            // Evening routine
            routineHTML += '<div style="background: #f3e5f5; padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 5px solid #9c27b0;"><h4 style="color: #9c27b0; margin-bottom: 20px;">Evening Routine (7 minutes)</h4>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 1: Double Cleanse</h5><p style="margin-bottom: 10px;"><strong>Remove the day:</strong> First oil cleanser (if wearing SPF), then gentle water-based cleanser.</p><p style="color: #666; font-size: 0.9rem;"><em>Optional: Micellar water for makeup/SPF removal</em></p></div>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 2: Active Treatment</h5><p style="margin-bottom: 10px;"><strong>Targeted care:</strong> Apply retinoid or stronger acne treatment (alternate nights if sensitive).</p><p style="color: #666; font-size: 0.9rem;"><em>Start slow: 2-3 times per week, build up gradually</em></p></div>';

            routineHTML += '<div class="ingredient-box"><h5 style="color: #7CB342; margin-bottom: 10px;">Step 3: Moisturize</h5><p style="margin-bottom: 10px;"><strong>Hydrate & repair:</strong> Use a ' + (skinType === 'oily' ? 'lightweight' : skinType === 'dry' ? 'rich' : 'balanced') + ' moisturizer to support skin barrier.</p><p style="color: #666; font-size: 0.9rem;"><em>Look for: Hyaluronic acid, ceramides, peptides</em></p></div>';

            routineHTML += '</div>';

            // Additional tips based on concerns
            if (concerns.length > 0) {
                routineHTML += '<div style="background: #e8f5e8; padding: 20px; border-radius: 15px; margin: 20px 0;"><h4 style="color: #7CB342; margin-bottom: 15px;">Bonus Tips for Your Goals</h4>';
                concerns.forEach(function(concern) {
                    var tips = {
                        'hyperpigmentation': 'Add vitamin C serum in the morning for brightening',
                        'pores': 'Use BHA (salicylic acid) 2-3 times per week',
                        'wrinkles': 'Consistent retinoid use will help with fine lines',
                        'dullness': 'Weekly gentle exfoliation can restore glow',
                        'sensitivity': 'Stick to fragrance-free, gentle products',
                        'dryness': 'Layer a hydrating toner before moisturizer',
                        'oiliness': 'Use oil-absorbing products and lightweight formulas',
                        'texture': 'Gentle chemical exfoliation 1-2 times per week'
                    };
                    if (tips[concern]) {
                        routineHTML += '<p style="margin: 5px 0; color: #666;">• ' + tips[concern] + '</p>';
                    }
                });
                routineHTML += '</div>';
            }

            // Important disclaimer
            routineHTML += '<div style="background: #ffebee; padding: 20px; border-radius: 15px; margin: 20px 0; border-left: 4px solid #f44336;"><h4 style="color: #f44336; margin-bottom: 10px;">Important Disclaimer</h4><p style="color: #666; line-height: 1.6; margin: 0;">This routine is based on AI analysis and general skincare principles. Results may vary. If you have persistent acne, sensitive skin, or skin conditions, please consult a dermatologist for personalized medical advice.</p></div>';

            // Success message
            routineHTML += '<div style="text-align: center; margin: 30px 0; padding: 25px; background: linear-gradient(135deg, #7CB342, #FF7A7A); color: white; border-radius: 15px;"><h4 style="margin-bottom: 15px;">Congratulations!</h4><p style="font-size: 1.1rem; line-height: 1.6;">You now have a science-backed routine personalized for your skin! Remember: consistency is key, and results typically show in 4-6 weeks.</p></div>';

            document.getElementById('routineContent').innerHTML = routineHTML;
        }

        function saveRoutineToEmail() {
            var email = document.getElementById('emailInput').value;
            if (!email || !email.includes('@')) {
                showError('Please enter a valid email address');
                return;
            }
            
            userEmail = email;
            showError('Email feature coming soon! Your routine has been saved locally for now.');
        }

        function saveToProfile() {
            localStorage.setItem('mrAcneSkinProfile', JSON.stringify(skinProfile));
            localStorage.setItem('mrAcneAnalysisResult', JSON.stringify(analysisResult));
            showError('Profile saved locally! Feature will be enhanced soon.');
        }

        function shareRoutine() {
            if (navigator.share) {
                navigator.share({
                    title: 'My Mr. Acne AI Analysis Results',
                    text: 'I just got my personalized skincare routine from Mr. Acne AI! Check it out:',
                    url: window.location.href
                });
            } else {
                // Fallback for browsers that don't support Web Share API
                var shareText = 'I just got my personalized skincare routine from Mr. Acne AI! Check it out: ' + window.location.href;
                navigator.clipboard.writeText(shareText).then(function() {
                    showError('Share link copied to clipboard!');
                }).catch(function() {
                    showError('Unable to copy share link. Please copy the URL manually.');
                });
            }
        }

        function restartAssessment() {
            // Reset all variables
            currentStep = 0;
            stepHistory = [0];
            selectedFile = null;
            analysisResult = null;
            skinProfile = {};
            selectedConcerns = [];
            completedTests = [];
            testResults = {};
            currentTestType = null;
            
            // Clear UI elements
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('analysisResults').innerHTML = '';
            var analyzeBtn = document.getElementById('analyzeBtn');
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }
            
            // Reset file input
            var fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.value = '';
            }
            
            // Reset upload area
            var uploadArea = document.getElementById('uploadArea');
            if (uploadArea) {
                uploadArea.style.borderColor = '#7CB342';
                uploadArea.style.background = '#FDF5E6';
            }
            
            // Clear all checkboxes
            var checkboxes = document.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(function(cb) {
                cb.checked = false;
            });
            
            // Go back to start
            nextStep(0);
            updateProgress();
        }

        // Handle browser navigation (back button)
        window.addEventListener('popstate', function(event) {
            if (stepHistory.length > 1) {
                previousStep();
            }
        });

        // Handle window resize for mobile responsiveness
        window.addEventListener('resize', function() {
            updateProgress();
        });

        // Keyboard navigation support
        document.addEventListener('keydown', function(event) {
            // Allow Enter to trigger buttons with focus
            if (event.key === 'Enter') {
                var focusedElement = document.activeElement;
                if (focusedElement && focusedElement.classList.contains('btn')) {
                    focusedElement.click();
                }
            }
            
            // Allow Escape to close modals
            if (event.key === 'Escape') {
                closeSkipModal();
            }
        });

        console.log('Mr. Acne application fully initialized and ready!');
    </script>
</body>
</html>
'''

# Run the application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, request, jsonify
import base64
import io
import os
import traceback
from PIL import Image
import numpy as np
import cv2
import pytesseract  # For text recognition
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = Flask(__name__)

# Set the Tesseract path if needed (uncomment if Tesseract is not in your PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request to /predict endpoint")
        
        # Check if request has JSON data
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        print(f"Received data with keys: {data.keys()}")
        
        if 'image' not in data or 'mode' not in data:
            print("Error: Missing required parameters")
            return jsonify({'error': 'Missing image or mode parameter'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            print("Successfully decoded base64 image")
        except Exception as e:
            print(f"Error decoding base64: {str(e)}")
            return jsonify({'error': f'Invalid base64 encoding: {str(e)}'}), 400
        
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"Opened image: {image.format}, size: {image.size}, mode: {image.mode}")
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        try:
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            print(f"Converted to OpenCV image, shape: {opencv_image.shape}")
        except Exception as e:
            print(f"Error converting to OpenCV format: {str(e)}")
            return jsonify({'error': f'Image conversion error: {str(e)}'}), 400
        
        mode = data['mode'].lower()
        print(f"Processing in mode: {mode}")
        result = ""
        
        if mode == 'text':
            try:
                # Text recognition using Tesseract OCR
                print("Performing OCR with Tesseract")
                result = pytesseract.image_to_string(opencv_image)
                if not result.strip():
                    result = "No text detected in image"
                print(f"OCR result: {result[:50]}..." if len(result) > 50 else f"OCR result: {result}")
            except Exception as e:
                print(f"Error in text recognition: {str(e)}")
                traceback.print_exc()
                return jsonify({'error': f'Text recognition error: {str(e)}'}), 500
        
        elif mode == 'scene':
            # For demo purposes, just return a placeholder
            result = "This image appears to show a scene with various objects (placeholder response)"
            print("Returning placeholder scene description")
        
        else:
            print(f"Invalid mode requested: {mode}")
            return jsonify({'error': 'Invalid mode. Use "text" or "scene"'}), 400
        
        print("Successfully processed request, returning result")
        return jsonify({'prediction': result})
    
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return "Vision AI API is running. Use POST /predict endpoint with image and mode parameters."

if __name__ == '__main__':
    print("Starting Vision AI API server...")
    # Check if tesseract is available
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"WARNING: Tesseract may not be installed or configured correctly: {str(e)}")
        print("Text recognition mode may not work!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
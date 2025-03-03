import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import logging
import time
from PIL import Image
import sys
from threading import Thread, Lock
from queue import Queue
import base64
import io
import numpy as np
from flask import Flask, jsonify, request, Response
import json
from flask_cors import CORS
import socket
import os
def setup_logging():
    """Configure logging with basic formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device
        self.current_caption = f"Initializing caption... ({device.upper()})"
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _caption_worker(self):
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)
                    with self.lock:
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)  # Prevent busy waiting

    def _generate_caption(self, image):
        try:
            # Resize to 640x480 (or any other size)
            image_resized = cv2.resize(image, (640, 480))

            # Convert to RGB
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process the image for captioning
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1
                )

            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return f"BLIP: {caption} ({self.device.upper()})"
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return f"BLIP: Caption generation failed ({self.device.upper()})"

    def update_frame(self, frame):
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except:
                pass  # Queue is full, skip this frame

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)  # added timeout coz it was hanging 

def get_gpu_usage():
    """Get the GPU memory usage and approximate utilization"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB

        memory_used_percent = (memory_allocated / memory_total) * 100
        gpu_info = f"GPU Memory Usage: {memory_used_percent:.2f}% | Allocated: {memory_allocated:.2f} MB / {memory_total:.2f} MB"
        
        return gpu_info
    else:
        return "GPU not available"

def get_local_ip():
    try:
        # Create a socket and connect to an external server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Could not determine IP"

def load_models():
    """Load BLIP model"""
    try:
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            # Set GPU memory usage limit to 90%
            torch.cuda.set_per_process_memory_fraction(0.9)
            blip_model = blip_model.to('cuda')

        return blip_processor, blip_model, device
    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        return None, None, None

# ====== API CODE WITH CAPTIONING ======
# Initialize Flask app for the API
app = Flask(__name__)
CORS(app)  # CORS for cross origin requests from Flutter

# Global variables for API
stream_fps = 0
prev_frame_time = 0
caption_generator = None
current_frame = None
frame_lock = Lock()
processed_frames_queue = Queue(maxsize=5)  # Queue for processed frames with captions

def setup_captioning_stream():
    """Initialize resources for streaming with captioning"""
    global caption_generator
    
    logger.info("Loading BLIP model for streaming with captioning...")
    blip_processor, blip_model, device = load_models()
    if None in (blip_processor, blip_model):
        logger.error("Failed to load the BLIP model. Falling back to pure streaming.")
        return False
    
    logger.info(f"Using {device.upper()} for inference in API mode.")
    caption_generator = CaptionGenerator(blip_processor, blip_model, device)
    return True

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'mode': 'Video streaming with captioning' if caption_generator else 'Pure video streaming'
    })

@app.route('/api/caption', methods=['GET'])
def get_caption():
    """Endpoint to get the current caption"""
    global caption_generator
    
    if caption_generator:
        return jsonify({
            'caption': caption_generator.get_caption(),
            'timestamp': time.time()
        })
    else:
        return jsonify({
            'caption': 'Captioning not available',
            'timestamp': time.time()
        })

@app.route('/api/stream', methods=['GET'])
def video_feed():
    """Stream processed frames back to the mobile app"""
    def generate_frames():
        """Generate frames from the processed frames queue"""
        global processed_frames_queue
        
        while True:
            if not processed_frames_queue.empty():
                frame = processed_frames_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # If no frame is available, send a placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera feed...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)  # Wait a bit before sending another placeholder
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/send_frame', methods=['POST'])
def receive_frame():
    """Receive a frame from the mobile app for processing"""
    global caption_generator, processed_frames_queue, stream_fps, prev_frame_time
    
    try:
        # Get the image data from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400
            
        file = request.files['image']
        image_bytes = file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Calculate FPS
        current_time = time.time()
        if prev_frame_time > 0:
            stream_fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        
        # Process the frame with caption generator
        if caption_generator:
            caption_generator.update_frame(frame.copy())
            current_caption = caption_generator.get_caption()
            
            # Add caption to frame
            max_width = 40
            caption_lines = [current_caption[i:i + max_width] for i in range(0, len(current_caption), max_width)]
            
            y_offset = 40
            for line in caption_lines:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Add FPS info
            cv2.putText(frame, f"FPS: {stream_fps:.2f}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add to processed frames queue for streaming back
            if processed_frames_queue.full():
                try:
                    processed_frames_queue.get_nowait()  # Remove oldest frame if queue is full
                except:
                    pass
            processed_frames_queue.put(frame)
            
            # Return the caption as JSON response
            return jsonify({
                'success': True,
                'caption': current_caption
            }), 200
        else:
            # Without captioning, just return success
            return jsonify({'success': True, 'caption': 'Captioning not available'}), 200
        
    except Exception as e:
        logger.error(f"Error receiving frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream_info', methods=['GET'])
def stream_info():
    """Return current streaming information"""
    global caption_generator, stream_fps
    
    info = {
        'fps': stream_fps,
        'timestamp': time.time()
    }
    
    if caption_generator:
        info['caption'] = caption_generator.get_caption()
        info['captioning_enabled'] = True
    else:
        info['captioning_enabled'] = False
    
    return jsonify(info)

def run_api(host='0.0.0.0', port=5000, with_captioning=True):
    """Run the Flask API server"""
    local_ip = get_local_ip()
    
    if with_captioning:
        logger.info("Setting up streaming API with captioning...")
        setup_captioning_stream()
    else:
        logger.info("Setting up pure video streaming without image processing")
    
    # Print the IP in a format that's easy to copy into Flutter
    print("\n====================")
    print(f"API_BASE_URL = 'http://{local_ip}:{port}'")  # << Copy this into Flutter
    print("====================\n")
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Server accessible at: http://{local_ip}:{port}")
    logger.info(f"Stream endpoint: http://{local_ip}:{port}/api/stream")
    logger.info(f"Caption endpoint: http://{local_ip}:{port}/api/caption")
    
    app.run(host=host, port=port, debug=False, threaded=True)

# Modified main function to support both local streaming and API mode
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run_api(host='0.0.0.0', port=port, with_captioning=True)
    logger = setup_logging()
    
    # Default to API mode with captioning
    with_captioning = True
    
    # Check command line arguments for mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-captioning":
            with_captioning = False
        elif sys.argv[1] == "--help":
            print("Usage: python app.py [--no-captioning]")
            print("  --no-captioning   Run API without captioning (pure streaming)")
            print("  --help            Show this help message")
            sys.exit(0)
            
    # Run API server
    run_api(with_captioning=with_captioning)
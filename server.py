from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import subprocess
import os
from flask_cors import CORS

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Folder to store processing videos
PROCESSING_FOLDER = 'processing_videos'

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join(PROCESSING_FOLDER, video.filename)

    try:
        video.save(video_path)
        print(f"✅ Video saved at: {video_path}")

        # Run the processing script in the background
        socketio.start_background_task(process_video, video_path, emit_progress)
        
        
        return jsonify({'message': 'Video is being processed.'})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

# Background task for processing video
def process_video(video_path, send_progress_callback):
    try:
        send_progress_callback("Starting video processing...", 0)

        result = subprocess.run(
            ['python', 'main.py', video_path], capture_output=True, text=True
        )

        if result.returncode != 0:
            send_progress_callback("Processing failed", 0)
            print(f"⚠️ Errors: {result.stderr}")
        else:
            send_progress_callback("Processing Complete!", 100)

    except Exception as e:
        send_progress_callback(f"Error occurred during processing: {str(e)}", 0)
        print(f"❌ Error: {e}")

# Emit progress updates to the frontend
def emit_progress(message, percent_complete):
    socketio.emit('progress', {'message': message, 'progress': percent_complete})

if __name__ == '__main__':
    print("🚀 Starting server at http://0.0.0.0:3000")
    socketio.run(app, host='0.0.0.0', port=3000, debug=True)

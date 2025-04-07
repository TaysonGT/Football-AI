from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import subprocess
import os
from flask_cors import CORS
import shutil

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Video storage paths
INPUT_VIDEOS_FOLDER = 'input_videos'
PROCESSING_FOLDER = 'processing_videos'
OUTPUT_FOLDER = 'output_videos'

# Ensure folders exist
os.makedirs(INPUT_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PROCESSING_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_video():
    # Construct full paths
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join(PROCESSING_FOLDER, video.filename)

    try:
        video.save(video_path)
        print(f"✅ Video saved at: {video_path}")

        # Start processing
        socketio.start_background_task(process_video_task, video_path, emit_progress)
        return jsonify({
            'message': 'Video processing started',
            'input_path': video_path,
            'output_path': os.path.join(OUTPUT_FOLDER, f'processed_{video.name}')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_task(video_path, emit_callback):
    proc = subprocess.Popen(
        ['python', 'main.py', video_path],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    for line in proc.stdout:
        if line.startswith("PROGRESS:"):
            _, pct, msg = line.split(":", 2)
            emit_callback(msg.strip(), float(pct))

# server.py
def emit_progress(message, percent):
    print(f"DEBUG: Emitting progress - {message} ({percent}%)")  # Debug log
    socketio.emit('progress', {
        'message': message,
        'progress': percent
    }, namespace='/progress')  # Explicit namespace helps debugging

if __name__ == '__main__':
    print(f"🔍 Watching for videos in: {INPUT_VIDEOS_FOLDER}")
    print("🚀 Server ready at http://0.0.0.0:3000")
    socketio.run(app, host='0.0.0.0', port=3000)

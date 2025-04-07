from flask import Flask, request, jsonify
from flask_socketio import SocketIO, join_room
import subprocess
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Update your Socket.IO initialization:
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   logger=True,  # Enable debugging
                   async_mode='gevent',
                   engineio_logger=True)  # More verbose logs

# Video storage paths
INPUT_VIDEOS_FOLDER = 'input_videos'
PROCESSING_FOLDER = 'processing_videos'
OUTPUT_FOLDER = 'output_videos'

# Ensure folders exist
os.makedirs(INPUT_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PROCESSING_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['POST'])
def home():
    return "Server is running"

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

        sid = request.form.get('socket_id')
        if not sid:
            return jsonify({'error': 'Socket ID not provided'}), 400

        # Start processing
        socketio.start_background_task(process_video_task, video_path, sid=sid)
        return jsonify({
            'message': 'Video processing started',
            'input_path': video_path,
            'output_path': os.path.join(OUTPUT_FOLDER, f'processed_{video.name}')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_task(video_path, sid):
    proc = subprocess.Popen(
        ['python', 'main.py', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
            
        if output and output.startswith("PROGRESS:"):
            try:
                _, pct, msg = output.strip().split(":", 2)
                # Add room targeting
                socketio.emit('updates', {
                    'message': msg.strip(),
                    'progress': float(pct)
                }, room=sid)  # Critical change
            except ValueError as e:
                print(f"Progress parse error: {e}")
    if proc.returncode != 0:
        error = proc.stderr.read()
        socketio.emit('error', {'message': error}, room=sid)

# Add to server.py
@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    print(f"Client joined room: {room}")

if __name__ == '__main__':
    print(f"🔍 Watching for videos in: {INPUT_VIDEOS_FOLDER}")
    print("🚀 Server ready at http://0.0.0.0:3000")
    socketio.run(app, host='0.0.0.0', port=3000, debug=True)

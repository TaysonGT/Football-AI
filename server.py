from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import subprocess

app = Flask(__name__)
CORS(app)

PROCESSING_FOLDER = 'processing_videos'
OUTPUT_FOLDER = 'output_videos'
os.makedirs(PROCESSING_FOLDER, exist_ok=True)  # Ensure output folder exists


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_path = os.path.join(PROCESSING_FOLDER, video.filename)
    video_name, _ = os.path.splitext(video.filename)
    try:
            video.save(video_path)  
            print(f"✅ Video saved at: {video_path}")

            # Run processing script
            result = subprocess.run(['python', 'main.py', video_path], capture_output=True, text=True)

            # Debugging: Print subprocess output
            print(f"🖥️ Subprocess Output:\n{result.stdout}")
            print(f"⚠️ Errors (if any):\n{result.stderr}")

            if result.returncode != 0:
                return jsonify({'error': 'Processing failed', 'details': result.stderr}), 500

            # Check if output exists
            if not os.path.exists(video_path):
                return jsonify({'error': 'Processed video not found'}), 500

            return jsonify({'message': 'Processing complete', 'output': f"processed_{video_name}.mp4"})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500


@app.route('/open_video/<filename>', methods=['GET'])
def open_video(filename):
    video_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        if os.name == 'nt':  # Windows
            os.startfile(video_path)
        elif os.name == 'posix':  # Mac/Linux
            subprocess.run(['open' if os.uname().sysname == "Darwin" else 'xdg-open', video_path])

        return jsonify({'message': 'Video opened successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/videos', methods=['GET'])
def list_videos():
    """Returns a list of all videos in the output folder"""
    videos = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        return jsonify({'message': 'No videos found'}), 404

    return jsonify({'videos': videos})


if __name__ == '__main__':
    app.run(debug=True)


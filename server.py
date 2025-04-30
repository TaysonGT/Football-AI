import subprocess
import os
import asyncio
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import functools
import shutil

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

connections = {}

# Video storage paths
OUTPUT_FOLDER = 'output_videos'
UPLOAD_FOLDER = 'uploads'

# Ensure folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, video.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"video_path": file_path}

def run_subprocess_sync(video_path):
    # Start subprocess and stream logs
    proc = subprocess.Popen(
        ["python", "main.py", video_path],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        text=True
    )
    for line in iter(proc.stdout.readline, ''):
        yield line.strip()
    proc.stdout.close()
    proc.wait()

@app.websocket('/ws/process')
async def process_video(websocket:WebSocket):
    await websocket.accept()
    print("🟢 WebSocket accepted")
    data = await websocket.receive_json()
    video_path = data.get("video_path")

    if not os.path.exists(video_path):
        await websocket.send_json({"error": "Video not found"})
        return
    loop = asyncio.get_running_loop()
    for line in await loop.run_in_executor(None, functools.partial(run_subprocess_sync, video_path)):
        if line.startswith("PROGRESS:"):
            _, pct, msg = line.split(":", 2)
            await websocket.send_json({
                "progress": float(pct),
                "message": msg.strip()
            })
        elif line.startswith("FinalPossession:"):
            _, t1, t2, c1, c2 = line.split(":", 4)
            await websocket.send_json({
                "finalPossession": {
                    "team1": {
                        "percent": float(t1),
                        "color": c1.strip()
                    },
                    "team2": {
                        "percent": float(t2),
                        "color": c2.strip()
                    }
                }
            })
        else:
            await websocket.send_json({
                "log": line,
            })
    await websocket.send_json({"done": True})

# Run FastAPI server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
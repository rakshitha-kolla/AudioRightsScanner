import logging
import os
import json
import uuid
import threading
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .services.service import ACRCloudService
from .config import ACR_ACCESS_KEY, ACR_ACCESS_SECRET, ACR_HOST, UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS
from .utils import FileHandler, ResultHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YAMNET_AVAILABLE = False
try:
    import tflite_runtime.interpreter as _tflite
    import librosa
    YAMNET_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as _tflite
        import librosa
        YAMNET_AVAILABLE = True
    except ImportError:
        pass

app = FastAPI(title="Audio Copyright Detector", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    acr_service = ACRCloudService(ACR_ACCESS_KEY, ACR_ACCESS_SECRET, ACR_HOST)
except Exception as e:
    logger.error(f"Failed to initialize ACRCloud: {e}")

FileHandler.ensure_folders()

yamnet_detector_instance = None

@app.on_event("startup")
async def preload_yamnet():
    global yamnet_detector_instance
    if not YAMNET_AVAILABLE:
        return

    def _load():
        global yamnet_detector_instance
        try:
            from .services.yamnet_detector import YAMNetDetector
            detector = YAMNetDetector()
            detector.load_model()
            yamnet_detector_instance = detector
            logger.info("YAMNet model ready")
        except Exception as e:
            logger.warning(f"YAMNet preload failed: {e}")

    threading.Thread(target=_load, daemon=True).start()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "acr_cloud": "configured" if (ACR_ACCESS_KEY and ACR_ACCESS_SECRET) else "not_configured",
        "yamnet_available": YAMNET_AVAILABLE,
        "version": "3.0.0"
    }

jobs = {}

def run_detection_job(job_id, file_path, filename):
    try:
        jobs[job_id]["status"] = "processing"

        if YAMNET_AVAILABLE:
            waited = 0
            while yamnet_detector_instance is None and waited < 60:
                time.sleep(3)
                waited += 3
            detection_result = acr_service.identify_with_yamnet(
                file_path, detector_instance=yamnet_detector_instance
            )
            detection_result["detection_method"] = "yamnet"
        else:
            detection_result = acr_service.identify_with_timeline(file_path)
            detection_result["detection_method"] = "timeline"

        result_file = ResultHandler.save_result(filename, detection_result)
        detection_result["result_file"] = result_file
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = detection_result

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/detect")
async def detect_from_file(audio_file: UploadFile = File(...)):
    file_ext = audio_file.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)

    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    threading.Thread(target=run_detection_job, args=(job_id, file_path, audio_file.filename), daemon=True).start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/detect/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/results")
async def get_results():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    results = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith('.json'):
            with open(os.path.join(RESULTS_FOLDER, filename)) as f:
                result = json.load(f)
                results.append({
                    "filename": filename,
                    "file_name": result.get('file_name'),
                    "copyrighted": result.get('copyrighted'),
                    "timestamp": result.get('timestamp'),
                })
    return {"total": len(results), "results": results}


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    result_path = os.path.join(RESULTS_FOLDER, result_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(result_path) as f:
        return json.load(f)


@app.get("/api/stats")
async def get_statistics():
    upload_count = len(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else 0
    result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')] if os.path.exists(RESULTS_FOLDER) else []
    copyrighted_count = 0
    for filename in result_files:
        try:
            with open(os.path.join(RESULTS_FOLDER, filename)) as f:
                if json.load(f).get('copyrighted'):
                    copyrighted_count += 1
        except Exception:
            pass
    return {
        "total_uploads": upload_count,
        "total_detections": len(result_files),
        "copyrighted_found": copyrighted_count,
        "non_copyrighted": len(result_files) - copyrighted_count,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(file_path)
    return {"message": f"File {filename} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
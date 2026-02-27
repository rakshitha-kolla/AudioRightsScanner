from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
import json
from datetime import datetime
from pathlib import Path

from service import ACRCloudService
from config import ACR_ACCESS_KEY, ACR_ACCESS_SECRET, ACR_HOST, UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS
from utils import FileHandler, ResultHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if YAMNet dependencies are available
YAMNET_AVAILABLE = False
try:
    import tflite_runtime.interpreter as _tflite
    import librosa
    YAMNET_AVAILABLE = True
    logger.info(" YAMNet TFLite dependencies available")
except ImportError:
    try:
        import tensorflow.lite as _tflite
        import librosa
        YAMNET_AVAILABLE = True
        logger.info("YAMNet TFLite (via tensorflow) available")
    except ImportError as _err:
        logger.warning(f" YAMNet not available ({_err}). Falling back to timeline chunking.")

# Initialize FastAPI
app = FastAPI(
    title=" Audio Copyright Detector",
    description="Detect copyrighted music using ACRCloud API",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
try:
    acr_service = ACRCloudService(ACR_ACCESS_KEY, ACR_ACCESS_SECRET, ACR_HOST)
    logger.info("ACRCloud service initialized")
except Exception as e:
    logger.error(f" Failed to initialize ACRCloud service: {e}")

# Ensure folders exist
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
            logger.info("Pre-loading YAMNet model in background...")
            from yamnet_detector import YAMNetDetector
            detector = YAMNetDetector()
            detector.load_model()
            yamnet_detector_instance = detector
            logger.info("YAMNet model ready")
        except Exception as e:
            logger.warning(f" YAMNet preload failed: {e}")

    import threading
    threading.Thread(target=_load, daemon=True).start()

app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "acr_cloud": "configured" if (ACR_ACCESS_KEY and ACR_ACCESS_SECRET) else "not_configured",
        "yamnet_available": YAMNET_AVAILABLE,
        "version": "3.0.0"
    }

import uuid
import threading

jobs = {}

def run_detection_job(job_id: str, file_path: str, filename: str):
    try:
        jobs[job_id]["status"] = "processing"

        if YAMNET_AVAILABLE:
            logger.info("Using YAMNet + Chroma + ACRCloud pipeline")
            import time
            waited = 0
            while yamnet_detector_instance is None and waited < 60:
                logger.info("Waiting for YAMNet to finish loading...")
                time.sleep(3)
                waited += 3
            detection_result = acr_service.identify_with_yamnet(
                file_path,
                detector_instance=yamnet_detector_instance
            )
            detection_result["detection_method"] = "yamnet"
        else:
            logger.info("Using timeline chunking")
            detection_result = acr_service.identify_with_timeline(file_path)
            detection_result["detection_method"] = "timeline"

        result_file = ResultHandler.save_result(filename, detection_result)
        detection_result["result_file"] = result_file

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = detection_result
        logger.info(f" Job {job_id} complete")

    except Exception as e:
        logger.error(f" Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


@app.post("/api/detect")
async def detect_from_file(audio_file: UploadFile = File(...)):
    try:
        logger.info(f"Received upload: {audio_file.filename}")

        file_ext = audio_file.filename.split('.')[-1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)

        with open(file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        logger.info(f"File saved: {file_path}")

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued", "result": None, "error": None}

        thread = threading.Thread(
            target=run_detection_job,
            args=(job_id, file_path, audio_file.filename),
            daemon=True
        )
        thread.start()

        logger.info(f"Job {job_id} started")
        return {"job_id": job_id, "status": "queued"}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f" Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detect/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/api/detect/yamnet")
async def detect_yamnet_only(
    audio_file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    chroma_threshold: float = 0.35
):
    if not YAMNET_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="YAMNet not available. Install: pip install tensorflow tensorflow-hub librosa"
        )

    file_ext = audio_file.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)

    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    result = acr_service.identify_with_yamnet(
        file_path,
        confidence_threshold=confidence_threshold,
        chroma_threshold=chroma_threshold
    )
    result["detection_method"] = "yamnet"

    result_file = ResultHandler.save_result(audio_file.filename, result)
    result["result_file"] = result_file

    return result


@app.post("/api/detect/timeline")
async def detect_timeline_only(audio_file: UploadFile = File(...)):
    file_ext = audio_file.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)

    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    result = acr_service.identify_with_timeline(file_path)
    result["detection_method"] = "timeline"

    result_file = ResultHandler.save_result(audio_file.filename, result)
    result["result_file"] = result_file

    return result


@app.post("/api/detect-url")
async def detect_from_path(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    is_valid, message = FileHandler.validate_audio_file(file_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    detection_result = acr_service.identify(file_path)
    result_file = ResultHandler.save_result(file_path, detection_result)
    detection_result["result_file"] = result_file
    return detection_result


@app.post("/api/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    results = []

    for audio_file in files:
        try:
            file_ext = audio_file.filename.split('.')[-1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                results.append({"file": audio_file.filename, "error": f"Invalid format"})
                continue

            file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            with open(file_path, "wb") as f:
                f.write(await audio_file.read())

            detection_result = acr_service.identify(file_path)
            result_file = ResultHandler.save_result(audio_file.filename, detection_result)
            detection_result["file"] = audio_file.filename
            detection_result["result_file"] = result_file
            results.append(detection_result)

        except Exception as e:
            results.append({"file": audio_file.filename, "error": str(e)})

    return {"results": results, "total": len(results), "processed_at": datetime.now().isoformat()}


@app.get("/api/results")
async def get_results():
    try:
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        results = []
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.endswith('.json'):
                file_path = os.path.join(RESULTS_FOLDER, filename)
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    results.append({
                        "filename": filename,
                        "file_name": result.get('file_name'),
                        "copyrighted": result.get('copyrighted'),
                        "timestamp": result.get('timestamp'),
                        "music": result.get('music')
                    })
        return {"total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    result_path = os.path.join(RESULTS_FOLDER, result_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(result_path, 'r') as f:
        return json.load(f)


@app.get("/api/results/{result_id}/download")
async def download_result(result_id: str):
    result_path = os.path.join(RESULTS_FOLDER, result_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(path=result_path, filename=result_id, media_type="application/json")


@app.get("/api/files")
async def list_files():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                files.append({
                    "name": filename,
                    "size": file_size,
                    "size_mb": round(file_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        return {"total": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(file_path)
    return {"message": f"File {filename} deleted successfully"}


@app.get("/api/stats")
async def get_statistics():
    try:
        upload_count = len([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]) if os.path.exists(UPLOAD_FOLDER) else 0
        result_count = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]) if os.path.exists(RESULTS_FOLDER) else 0

        copyrighted_count = 0
        if os.path.exists(RESULTS_FOLDER):
            for filename in os.listdir(RESULTS_FOLDER):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(RESULTS_FOLDER, filename), 'r') as f:
                            result = json.load(f)
                            if result.get('copyrighted') == True:
                                copyrighted_count += 1
                    except:
                        pass

        return {
            "total_uploads": upload_count,
            "total_detections": result_count,
            "copyrighted_found": copyrighted_count,
            "non_copyrighted": result_count - copyrighted_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
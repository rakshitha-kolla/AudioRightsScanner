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

from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="🎵 Audio Copyright Detector",
    description="Detect copyrighted music using ACRCloud API",
    version="2.0.0"
)

# Add CORS middleware (allow frontend access)
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
    logger.info("✓ ACRCloud service initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize ACRCloud service: {e}")

# Ensure folders exist
FileHandler.ensure_folders()

# ==================== HEALTH CHECK ====================


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    """Check API health and ACRCloud connection"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "acr_cloud": "configured" if (ACR_ACCESS_KEY and ACR_ACCESS_SECRET) else "not_configured",
        "version": "2.0.0"
    }


# ==================== DETECTION ENDPOINTS ====================

@app.post("/api/detect")
async def detect_from_file(audio_file: UploadFile = File(...)):
    """
    Detect copyrighted music from uploaded audio file
    
    Returns:
    - copyrighted: bool (True if copyrighted music detected)
    - music: dict (Title, Artist, Album if detected)
    - error: str (if error occurred)
    """
    try:
        # Validate file
        logger.info(f"📥 Received upload: {audio_file.filename}")
        
        file_ext = audio_file.filename.split('.')[-1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Create upload folder
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        with open(file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        logger.info(f"💾 File saved: {file_path}")
        
        # Run detection
        logger.info(f"🔍 Analyzing audio...")
        detection_result = acr_service.identify_with_timeline(file_path)
        
        # Save result
        result_file = ResultHandler.save_result(audio_file.filename, detection_result)
        logger.info(f"📁 Result saved: {result_file}")
        
        # Add result file path to response
        detection_result["result_file"] = result_file
        
        return detection_result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"✗ Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-url")
async def detect_from_path(file_path: str):
    """
    Detect copyrighted music from file path on server
    
    Request body:
    {
        "file_path": "/path/to/audio/file.mp3"
    }
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Validate format
        is_valid, message = FileHandler.validate_audio_file(file_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        logger.info(f"🔍 Analyzing: {file_path}")
        
        # Run detection
        detection_result = acr_service.identify(file_path)
        
        # Save result
        result_file = ResultHandler.save_result(file_path, detection_result)
        logger.info(f"📁 Result saved: {result_file}")
        
        # Add result file path to response
        detection_result["result_file"] = result_file
        
        return detection_result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"✗ Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== BATCH DETECTION ====================

@app.post("/api/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Detect copyright in multiple files (batch processing)
    
    Returns: List of detection results
    """
    results = []
    
    for audio_file in files:
        try:
            logger.info(f"📥 Processing: {audio_file.filename}")
            
            # Validate file
            file_ext = audio_file.filename.split('.')[-1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                results.append({
                    "file": audio_file.filename,
                    "error": f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            # Save file
            file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
            
            # Detect
            detection_result = acr_service.identify(file_path)
            
            # Save result
            result_file = ResultHandler.save_result(audio_file.filename, detection_result)
            
            detection_result["file"] = audio_file.filename
            detection_result["result_file"] = result_file
            results.append(detection_result)
        
        except Exception as e:
            logger.error(f"✗ Error processing {audio_file.filename}: {str(e)}")
            results.append({
                "file": audio_file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total": len(results), "processed_at": datetime.now().isoformat()}

# ==================== RESULTS ENDPOINTS ====================

@app.get("/api/results")
async def get_results():
    """List all saved detection results"""
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
        
        return {
            "total": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"✗ Error listing results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Get specific result by ID"""
    try:
        result_path = os.path.join(RESULTS_FOLDER, result_id)
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Result not found")
        
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        return result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"✗ Error retrieving result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{result_id}/download")
async def download_result(result_id: str):
    """Download result JSON file"""
    try:
        result_path = os.path.join(RESULTS_FOLDER, result_id)
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Result not found")
        
        return FileResponse(
            path=result_path,
            filename=result_id,
            media_type="application/json"
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"✗ Error downloading result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FILE ENDPOINTS ====================

@app.get("/api/files")
async def list_files():
    """List all uploaded audio files"""
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
        
        return {
            "total": len(files),
            "files": files
        }
    
    except Exception as e:
        logger.error(f"✗ Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete uploaded file"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        logger.info(f"✓ Deleted: {filename}")
        
        return {"message": f"File {filename} deleted successfully"}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"✗ Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STATISTICS ====================

@app.get("/api/stats")
async def get_statistics():
    """Get usage statistics"""
    try:
        # Count files
        upload_count = len([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]) if os.path.exists(UPLOAD_FOLDER) else 0
        
        # Count results
        result_count = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]) if os.path.exists(RESULTS_FOLDER) else 0
        
        # Count copyrighted
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
        logger.error(f"✗ Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
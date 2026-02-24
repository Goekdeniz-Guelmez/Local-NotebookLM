from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from enum import Enum
from typing import Optional
import tempfile
import os
import shutil
from pydantic import BaseModel
import uuid
from threading import Lock

# Import the new audio generator
from .processor import generate_audio

# Create FastAPI app
app = FastAPI(
    title="Audio Generator API",
    description="API for generating audios from PDF documents",
    version="2.0.0"
)

# Define enums for the choices matching generate_audio options
class FormatType(str, Enum):
    podcast = "podcast"
    narration = "narration"
    interview = "interview"
    panel_discussion = "panel-discussion"
    summary = "summary"
    article = "article"
    lecture = "lecture"
    q_and_a = "q-and-a"
    tutorial = "tutorial"
    debate = "debate"
    meeting = "meeting"
    analysis = "analysis"

class ContentLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"

class ContentStyle(str, Enum):
    normal = "normal"
    formal = "formal"
    casual = "casual"
    enthusiastic = "enthusiastic"
    serious = "serious"
    humorous = "humorous"
    gen_z = "gen-z"
    technical = "technical"

# Response models
class AudioResponse(BaseModel):
    job_id: str
    status: str
    message: str

class AudioStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    audio_url: Optional[str] = None

# Dictionary to store job statuses
job_status = {}
job_status_lock = Lock()


def set_job_status(job_id: str, data: dict) -> None:
    with job_status_lock:
        job_status[job_id] = data


def get_job_info(job_id: str) -> Optional[dict]:
    with job_status_lock:
        return job_status.get(job_id)

# Function to process audio in background using generate_audio
def process_audio(
    job_id: str,
    pdf_path: str,
    config_path: Optional[str] = None,
    format_type: FormatType = FormatType.summary,
    length: ContentLength = ContentLength.medium,
    style: ContentStyle = ContentStyle.normal,
    preference: Optional[str] = None,
    output_dir: str = "./output"
):
    job_output_dir = os.path.join(output_dir, job_id)
    try:
        audio_path = generate_audio(
            pdf_path=pdf_path,
            output_dir=job_output_dir,
            format_type=format_type.value,
            length=length.value,
            style=style.value,
            custom_preferences=preference,
        )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file was reported but not found at '{audio_path}'")

        set_job_status(
            job_id,
            {
                "status": "completed",
                "result": {"audio_path": audio_path},
                "audio_path": audio_path,
                "audio_url": f"/download-audio/{job_id}",
            },
        )
            
    except Exception as e:
        set_job_status(job_id, {"status": "failed", "error": str(e)})
    
    # Clean up the temporary files
    try:
        # Clean up input files
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if config_path and os.path.exists(config_path):
            os.remove(config_path)
            
    except Exception as e:
        print(f"Error cleaning up: {str(e)}")

@app.post("/generate-audio/", response_model=AudioResponse)
async def generate_audio_endpoint(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = None,
    format_type: FormatType = Form(FormatType.summary),
    length: ContentLength = Form(ContentLength.medium),
    style: ContentStyle = Form(ContentStyle.normal),
    preference: Optional[str] = Form(None),
    output_dir: str = Form("./output")
):
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create temp directory if it doesn't exist
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded PDF to a temporary file
    pdf_path = os.path.join(temp_dir, f"{job_id}_{pdf_file.filename}")
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)
    
    # Save the config file if provided
    config_path = None
    if config_file:
        config_path = os.path.join(temp_dir, f"{job_id}_{config_file.filename}")
        with open(config_path, "wb") as buffer:
            shutil.copyfileobj(config_file.file, buffer)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Update job status
    set_job_status(job_id, {"status": "processing"})
    
    # Add the task to background tasks
    background_tasks.add_task(
        process_audio,
        job_id=job_id,
        pdf_path=pdf_path,
        config_path=config_path,
        format_type=format_type,
        length=length,
        style=style,
        preference=preference,
        output_dir=output_dir
    )
    
    return AudioResponse(
        job_id=job_id,
        status="processing",
        message="Your audio generation job has been started"
    )

@app.get("/job-status/{job_id}", response_model=AudioStatusResponse)
async def get_job_status(job_id: str):
    job_info = get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return AudioStatusResponse(
        job_id=job_id,
        status=job_info["status"],
        result=job_info.get("result"),
        audio_url=job_info.get("audio_url")
    )

@app.get("/download-audio/{job_id}")
async def download_audio(job_id: str, background_tasks: BackgroundTasks):
    job_info = get_job_info(job_id)
    if job_info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    if "audio_path" not in job_info:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    audio_path = job_info["audio_path"]
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found on server")
    
    # Schedule file deletion after response is sent
    def delete_file_after_download():
        try:
            # Wait a bit to ensure file is fully sent
            import time
            time.sleep(60)  # Give 60 seconds buffer
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
    
    # Add deletion task to background tasks
    background_tasks.add_task(delete_file_after_download)
    
    return FileResponse(
        path=audio_path, 
        filename=f"audio_{job_id}.wav", 
        media_type="audio/wav"
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint with API information
@app.get("/")
async def root():
    return {
        "api": "Audio Generator",
        "version": "2.0.0",
        "endpoints": [
            {"path": "/generate-audio/", "method": "POST", "description": "Generate audio from PDF"},
            {"path": "/job-status/{job_id}", "method": "GET", "description": "Check status of a job"},
            {"path": "/download-audio/{job_id}", "method": "GET", "description": "Download the generated audio file"},
            {"path": "/health", "method": "GET", "description": "API health check"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
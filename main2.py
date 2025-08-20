from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import math
import os
import uuid
import random
from typing import List, Tuple, Optional, Dict, Any
import json

app = FastAPI(title="8-Point Anatomy Analysis API", version="1.0.0")

# CORS middleware - adjust origins as needed for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "./uploads_8point"
PROCESSED_FOLDER = "./processed_8point"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 8-point anatomical specifications
EIGHT_POINT_SPECS = [
    {'id': 1, 'name': 'C7_Neck', 'label': 'C7 Neck Point', 'color': (255, 0, 0)},
    {'id': 2, 'name': 'Left_Shoulder', 'label': 'Left Shoulder', 'color': (0, 255, 0)},
    {'id': 3, 'name': 'Right_Shoulder', 'label': 'Right Shoulder', 'color': (0, 255, 0)},
    {'id': 4, 'name': 'Left_Armpit', 'label': 'Left Armpit', 'color': (0, 0, 255)},
    {'id': 5, 'name': 'Right_Armpit', 'label': 'Right Armpit', 'color': (0, 0, 255)},
    {'id': 6, 'name': 'Left_Waist', 'label': 'Left Waist Hollow', 'color': (255, 255, 0)},
    {'id': 7, 'name': 'Right_Waist', 'label': 'Right Waist Hollow', 'color': (255, 255, 0)},
    {'id': 8, 'name': 'Sacrum_Point', 'label': 'Sacrum/Tailbone Point', 'color': (255, 0, 255)}
]

# ========== PYDANTIC MODELS ==========
class Point(BaseModel):
    x: int
    y: int

class EightPointAnalysisRequest(BaseModel):
    image_id: str
    points: List[Point]

class AnalysisResult(BaseModel):
    measurement: str
    value: float

class EightPointResponse(BaseModel):
    success: bool
    results: List[AnalysisResult]
    annotated_image_url: Optional[str] = None
    message: str = ""

class ImageUploadResponse(BaseModel):
    success: bool
    image_id: str
    filename: str
    image_url: str
    message: str = ""

# ========== ANALYSIS FUNCTIONS ==========
def calculate_shoulder_symmetry(left_shoulder: Tuple[int, int], right_shoulder: Tuple[int, int]) -> float:
    """
    Calculate shoulder symmetry/level difference.
    TODO: Implement actual calculation based on your methodology
    """
    # Placeholder: return random value for now
    y_diff = abs(left_shoulder[1] - right_shoulder[1])
    return random.uniform(0.5, 15.0)

def calculate_armpit_symmetry(left_armpit: Tuple[int, int], right_armpit: Tuple[int, int]) -> float:
    """
    Calculate armpit level symmetry.
    TODO: Implement actual calculation based on your methodology
    """
    return random.uniform(0.3, 12.0)

def calculate_waist_symmetry(left_waist: Tuple[int, int], right_waist: Tuple[int, int]) -> float:
    """
    Calculate waist hollow symmetry.
    TODO: Implement actual calculation based on your methodology
    """
    return random.uniform(0.2, 10.0)

def calculate_spinal_alignment(c7_neck: Tuple[int, int], sacrum: Tuple[int, int]) -> float:
    """
    Calculate spinal alignment deviation from vertical.
    TODO: Implement actual calculation based on your methodology
    """
    return random.uniform(0.1, 8.0)

def calculate_8_point_analysis(points: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Main analysis function that calculates all 8-point postural measurements.
    
    Args:
        points: List of 8 anatomical points in order:
               [C7, L_Shoulder, R_Shoulder, L_Armpit, R_Armpit, L_Waist, R_Waist, Sacrum]
    
    Returns:
        Dictionary containing all calculated measurements
    """
    if len(points) != 8:
        raise ValueError(f"Expected 8 points, got {len(points)}")
    
    # Extract points by anatomical location
    c7_neck = points[0]
    left_shoulder = points[1]
    right_shoulder = points[2]
    left_armpit = points[3]
    right_armpit = points[4]
    left_waist = points[5]
    right_waist = points[6]
    sacrum_point = points[7]
    
    # Calculate various measurements
    results = {
        'shoulder_symmetry': calculate_shoulder_symmetry(left_shoulder, right_shoulder),
        'armpit_symmetry': calculate_armpit_symmetry(left_armpit, right_armpit),
        'waist_symmetry': calculate_waist_symmetry(left_waist, right_waist),
        'spinal_alignment': calculate_spinal_alignment(c7_neck, sacrum_point),
        'overall_posture_score': random.uniform(60.0, 95.0),
        'left_right_balance': random.uniform(-5.0, 5.0),
        'upper_body_tilt': random.uniform(-3.0, 3.0),
        
        # Store original points for reference
        'anatomical_points': {
            'c7_neck': c7_neck,
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_armpit': left_armpit,
            'right_armpit': right_armpit,
            'left_waist': left_waist,
            'right_waist': right_waist,
            'sacrum_point': sacrum_point
        }
    }
    
    return results

def draw_8_point_annotations(img: np.ndarray, points: List[Tuple[int, int]], results: Dict[str, Any]) -> np.ndarray:
    """Draw the 8 anatomical points and analysis lines on the image."""
    annotated = img.copy()
    
    if len(points) != 8:
        return annotated
    
    # Draw anatomical points
    for i, (point, spec) in enumerate(zip(points, EIGHT_POINT_SPECS)):
        # Draw point circle
        cv2.circle(annotated, point, 8, spec['color'], -1)
        cv2.circle(annotated, point, 8, (255, 255, 255), 2)
        
        # Draw point number
        cv2.putText(annotated, str(i+1), (point[0]+12, point[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw analysis lines
    c7, l_shoulder, r_shoulder, l_armpit, r_armpit, l_waist, r_waist, sacrum = points
    
    # Shoulder line
    cv2.line(annotated, l_shoulder, r_shoulder, (0, 255, 0), 2)
    
    # Armpit line
    cv2.line(annotated, l_armpit, r_armpit, (0, 0, 255), 2)
    
    # Waist line
    cv2.line(annotated, l_waist, r_waist, (255, 255, 0), 2)
    
    # Spinal line (C7 to Sacrum)
    cv2.line(annotated, c7, sacrum, (255, 0, 255), 3)
    
    # Vertical reference line through C7
    cv2.line(annotated, (c7[0], 0), (c7[0], annotated.shape[0]), (128, 128, 128), 1)
    
    # Add results text
    measurements = [
        f"Shoulder: {results['shoulder_symmetry']:.1f}",
        f"Armpit: {results['armpit_symmetry']:.1f}", 
        f"Waist: {results['waist_symmetry']:.1f}",
        f"Spine: {results['spinal_alignment']:.1f}",
        f"Score: {results['overall_posture_score']:.1f}"
    ]
    
    y_offset = 30
    for i, text in enumerate(measurements):
        # Background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            annotated,
            (5, y_offset + i*30 - text_height - 5),
            (15 + text_width, y_offset + i*30 + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (10, y_offset + i*30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return annotated

# ========== API ENDPOINTS ==========

@app.post("/upload-image-8point", response_model=ImageUploadResponse)
async def upload_image_8point(file: UploadFile = File(...)):
    """
    Upload an image file for 8-point analysis and return its ID for later processing.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique ID for the image
        image_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        filename = f"{image_id}{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the uploaded file
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Verify the image can be loaded by OpenCV
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        return ImageUploadResponse(
            success=True,
            image_id=image_id,
            filename=file.filename or filename,
            image_url=f"/images-8point/{filename}",
            message="Image uploaded successfully for 8-point analysis"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-8-points", response_model=EightPointResponse)
async def analyze_8_points(request: EightPointAnalysisRequest):
    """
    Analyze 8-point anatomy based on the provided points and image.
    
    Expected JSON format:
    {
        "image_id": "uuid-string",
        "points": [
            {"x": 100, "y": 200},  // C7 Neck
            {"x": 80, "y": 250},   // Left Shoulder
            {"x": 120, "y": 250},  // Right Shoulder
            {"x": 75, "y": 300},   // Left Armpit
            {"x": 125, "y": 300},  // Right Armpit
            {"x": 85, "y": 400},   // Left Waist
            {"x": 115, "y": 400},  // Right Waist
            {"x": 100, "y": 500}   // Sacrum Point
        ]
    }
    """
    try:
        # Validate input
        if not request.image_id:
            raise HTTPException(status_code=400, detail="image_id is required")
        
        if not request.points:
            raise HTTPException(status_code=400, detail="points list cannot be empty")
        
        if len(request.points) != 8:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected exactly 8 points, got {len(request.points)}"
            )
        
        # Find the image file
        image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(request.image_id)]
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found. Please upload the image first.")
        
        # Load the image
        image_path = os.path.join(UPLOAD_FOLDER, image_files[0])
        img = cv2.imread(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not load the image file")
        
        # Convert points to tuples
        points_tuples = [(pt.x, pt.y) for pt in request.points]
        
        # Calculate 8-point analysis
        analysis_results = calculate_8_point_analysis(points_tuples)
        
        # Create annotated image
        annotated_img = draw_8_point_annotations(img, points_tuples, analysis_results)
        
        # Save annotated image
        annotated_filename = f"annotated_8point_{request.image_id}.jpg"
        annotated_path = os.path.join(PROCESSED_FOLDER, annotated_filename)
        success = cv2.imwrite(annotated_path, annotated_img)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save annotated image")
        
        # Prepare response
        results = [
            AnalysisResult(measurement="Shoulder Symmetry", value=round(analysis_results['shoulder_symmetry'], 1)),
            AnalysisResult(measurement="Armpit Symmetry", value=round(analysis_results['armpit_symmetry'], 1)),
            AnalysisResult(measurement="Waist Symmetry", value=round(analysis_results['waist_symmetry'], 1)),
            AnalysisResult(measurement="Spinal Alignment", value=round(analysis_results['spinal_alignment'], 1)),
            AnalysisResult(measurement="Posture Score", value=round(analysis_results['overall_posture_score'], 1)),
        ]
        
        return EightPointResponse(
            success=True,
            results=results,
            annotated_image_url=f"/processed-8point/{annotated_filename}",
            message=f"Successfully analyzed 8-point anatomy measurements"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing 8-point anatomy: {str(e)}")

# ========== UTILITY ENDPOINTS ==========

@app.get("/images-8point/{filename}")
async def get_uploaded_image_8point(filename: str):
    """Serve uploaded images for 8-point analysis."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.get("/processed-8point/{filename}")
async def get_processed_image_8point(filename: str):
    """Serve processed/annotated images from 8-point analysis."""
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(filepath)

@app.get("/8point-specs")
async def get_8point_specs():
    """
    Get the 8-point specifications for the frontend to know the point order.
    """
    return {
        "specs": EIGHT_POINT_SPECS,
        "total_points": 8,
        "order": "Points should be provided in order: C7 Neck, Left Shoulder, Right Shoulder, Left Armpit, Right Armpit, Left Waist, Right Waist, Sacrum Point"
    }

@app.get("/health-8point")
async def health_check_8point():
    """Health check endpoint for 8-point analysis."""
    return {
        "status": "healthy",
        "message": "8-Point Anatomy Analysis API is running",
        "version": "1.0.0"
    }

@app.get("/8point")
async def eight_point_root():
    """Root endpoint for 8-point analysis with API information."""
    return {
        "message": "8-Point Anatomy Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload-image-8point": "Upload an image file for 8-point analysis",
            "POST /analyze-8-points": "Analyze 8-point anatomy from coordinates",
            "GET /8point-specs": "Get 8-point specifications",
            "GET /images-8point/{filename}": "Get uploaded image",
            "GET /processed-8point/{filename}": "Get processed image",
            "GET /health-8point": "Health check"
        },
        "point_order": [spec['label'] for spec in EIGHT_POINT_SPECS]
    }

# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
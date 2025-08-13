from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import math
import os
import uuid
from typing import List, Tuple, Optional
import json

app = FastAPI(title="Anatomy Angle Calculator Backend", version="1.0.0")

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Render provides /tmp for temporary files
UPLOAD_FOLDER = "/tmp/uploads"
PROCESSED_FOLDER = "/tmp/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Angle specifications - same as your original notebook
ANGLE_SPECS = [
    ('Ear–Neck vs Vertical', 2),
    ('Shoulder–Neck vs Vertical', 2),
    ('Three‑Point Angle #1', 3),
    ('Three‑Point Angle #2', 3),
]

# ========== PYDANTIC MODELS ==========
class Point(BaseModel):
    x: int
    y: int

class AngleCalculationRequest(BaseModel):
    image_id: str
    points: List[Point]

class AngleResult(BaseModel):
    label: str
    angle: float
    angle_index: int

class CalculationResponse(BaseModel):
    success: bool
    results: List[AngleResult]
    annotated_image_url: Optional[str] = None
    message: str = ""

class ImageUploadResponse(BaseModel):
    success: bool
    image_id: str
    filename: str
    image_url: str
    message: str = ""

# ========== ANGLE CALCULATION FUNCTIONS ==========
def angle_two_points_vs_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Compute the angle between the line p1→p2 and the vertical line through p2.
    p1, p2: (x,y) tuples in image coordinates (y grows downward).
    Returns angle in degrees.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    
    # Calculate magnitude
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0
    
    # Vertical direction (pointing up) in image coords is (0, -1)
    # Compute dot product with vertical: dot = dx*0 + dy*(-1) = -dy
    cos_theta = (-dy) / mag
    cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to avoid floating point errors
    angle_deg = math.degrees(math.acos(cos_theta))
    
    return angle_deg

def angle_three_points(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
    """Compute angle at point b formed by a→b→c."""
    # Vectors BA and BC
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    
    # Dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    
    # Guard against zero vectors
    if n1 == 0 or n2 == 0:
        return 0.0
    
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def calculate_angles(points: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
    """
    Calculate all angles based on the points provided.
    Points should be provided in the order specified by ANGLE_SPECS.
    """
    computed_angles = []
    point_index = 0
    
    for angle_index, (label, req_pts) in enumerate(ANGLE_SPECS):
        # Check if we have enough points for this angle
        if point_index + req_pts > len(points):
            break
            
        # Extract points for this angle
        pts = points[point_index:point_index + req_pts]
        point_index += req_pts
        
        if req_pts == 2:
            p1, p2 = pts
            ang = angle_two_points_vs_vertical(p1, p2)
            
            # Special case: For the second angle (Shoulder-Neck), invert with (180 - θ)
            if angle_index == 1:
                ang = 180.0 - ang
                
        else:  # req_pts == 3
            a, b, c = pts
            ang = angle_three_points(a, b, c)
        
        computed_angles.append((label, ang))
    
    return computed_angles

def draw_annotations(img: np.ndarray, points: List[Tuple[int, int]], angles: List[Tuple[str, float]]) -> np.ndarray:
    """Draw points, lines, and angle annotations on the image."""
    annotated = img.copy()
    point_index = 0
    
    for angle_index, (label, req_pts) in enumerate(ANGLE_SPECS):
        if point_index + req_pts > len(points):
            break
            
        pts = points[point_index:point_index + req_pts]
        
        # Draw points
        for pt in pts:
            cv2.circle(annotated, pt, 8, (0, 255, 0), -1)  # Green filled circles
            cv2.circle(annotated, pt, 8, (255, 255, 255), 2)  # White border
        
        if req_pts == 2:
            p1, p2 = pts
            # Draw the line between points
            cv2.line(annotated, p1, p2, (255, 0, 0), 3)  # Blue line
            
            # Draw vertical reference line
            vert_len = 100
            if angle_index == 1:  # Shoulder-Neck: vertical down
                p_vert = (p2[0], p2[1] + vert_len)
            else:  # Ear-Neck: vertical up
                p_vert = (p2[0], p2[1] - vert_len)
            
            cv2.line(annotated, p2, p_vert, (0, 0, 255), 3)  # Red vertical line
            
        else:  # req_pts == 3
            a, b, c = pts
            # Draw the two arms of the angle
            cv2.line(annotated, a, b, (255, 0, 0), 3)  # Blue lines
            cv2.line(annotated, b, c, (255, 0, 0), 3)
        
        point_index += req_pts
    
    # Add angle text annotations
    for i, (label, angle) in enumerate(angles):
        # Background rectangle for text
        text = f"{label}: {angle:.1f}°"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Draw background rectangle
        cv2.rectangle(
            annotated,
            (5, 20 + 35 * i - text_height - 5),
            (15 + text_width, 25 + 35 * i + 5),
            (255, 255, 255),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (10, 20 + 35 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # Red text
            2
        )
    
    return annotated

# ========== API ENDPOINTS ==========

@app.post("/upload-image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file and return its ID for later processing.
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
            image_url=f"/images/{filename}",
            message="Image uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/calculate-angles", response_model=CalculationResponse)
async def calculate_angles_endpoint(request: AngleCalculationRequest):
    """
    Calculate angles based on the provided points and image.
    """
    try:
        # Validate input
        if not request.image_id:
            raise HTTPException(status_code=400, detail="image_id is required")
        
        if not request.points:
            raise HTTPException(status_code=400, detail="points list cannot be empty")
        
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
        
        # Validate we have the expected number of points
        total_required_points = sum(req_pts for _, req_pts in ANGLE_SPECS)
        if len(points_tuples) < total_required_points:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough points. Expected {total_required_points}, got {len(points_tuples)}"
            )
        
        # Calculate angles using your original logic
        angles = calculate_angles(points_tuples)
        
        # Create annotated image
        annotated_img = draw_annotations(img, points_tuples, angles)
        
        # Save annotated image
        annotated_filename = f"annotated_{request.image_id}.jpg"
        annotated_path = os.path.join(PROCESSED_FOLDER, annotated_filename)
        success = cv2.imwrite(annotated_path, annotated_img)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save annotated image")
        
        # Prepare response
        results = [
            AngleResult(label=label, angle=round(angle, 2), angle_index=i)
            for i, (label, angle) in enumerate(angles)
        ]
        
        return CalculationResponse(
            success=True,
            results=results,
            annotated_image_url=f"/processed/{annotated_filename}",
            message=f"Successfully calculated {len(results)} angles"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating angles: {str(e)}")

# ========== UTILITY ENDPOINTS ==========

@app.get("/images/{filename}")
async def get_uploaded_image(filename: str):
    """Serve uploaded images."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.get("/processed/{filename}")
async def get_processed_image(filename: str):
    """Serve processed/annotated images."""
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(filepath)

@app.get("/angle-specs")
async def get_angle_specs():
    """Get the angle specifications for the frontend."""
    return {
        "specs": [{"label": label, "points_required": points} for label, points in ANGLE_SPECS],
        "total_points": sum(points for _, points in ANGLE_SPECS),
        "order": "Points should be provided in order: Ear-Neck (2 pts), Shoulder-Neck (2 pts), Three-Point #1 (3 pts), Three-Point #2 (3 pts)"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Anatomy Angle Calculator Backend is running",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Anatomy Angle Calculator Backend API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload-image": "Upload an image file",
            "POST /calculate-angles": "Calculate angles from points",
            "GET /angle-specs": "Get angle specifications",
            "GET /images/{filename}": "Get uploaded image",
            "GET /processed/{filename}": "Get processed image",
            "GET /health": "Health check"
        }
    }

# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
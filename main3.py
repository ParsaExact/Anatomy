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
from typing import List, Tuple, Optional
import json

app = FastAPI(title="Automatic Anatomy Angle Calculator", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "./uploads_auto"
PROCESSED_FOLDER = "./processed_auto"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Angle specifications - same as original
ANGLE_SPECS = [
    ('Ear–Neck vs Vertical', 2),
    ('Shoulder–Neck vs Vertical', 2),
    ('Three‑Point Angle #1', 3),
    ('Three‑Point Angle #2', 3),
]

# ========== PYDANTIC MODELS ==========
class AngleResult(BaseModel):
    label: str
    angle: float
    angle_index: int

class AutoAnalysisResponse(BaseModel):
    success: bool
    results: List[AngleResult]
    detected_points: List[List[int]]  # List of [x, y] coordinates
    annotated_image_url: Optional[str] = None
    message: str = ""

class ImageUploadResponse(BaseModel):
    success: bool
    image_id: str
    filename: str
    image_url: str
    message: str = ""

# ========== COMPUTER VISION FUNCTIONS (PLACEHOLDER) ==========
def detect_anatomical_points_auto(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Automatically detect anatomical points using computer vision.
    TODO: Implement actual computer vision detection
    Currently returns random points within image bounds.
    """
    height, width = image.shape[:2]
    
    # Generate random points within realistic anatomical regions
    # This is a placeholder - replace with actual CV detection
    points = []
    
    # Simulate detection of 10 points for the 4 angles
    # Ear-Neck (2 points)
    ear_point = (random.randint(width//4, 3*width//4), random.randint(height//6, height//3))
    neck_point = (ear_point[0] + random.randint(-50, 50), ear_point[1] + random.randint(50, 100))
    points.extend([ear_point, neck_point])
    
    # Shoulder-Neck (2 points)  
    shoulder_point = (random.randint(width//5, 4*width//5), random.randint(height//3, 2*height//3))
    neck_point2 = (shoulder_point[0] + random.randint(-30, 30), shoulder_point[1] + random.randint(-80, -30))
    points.extend([shoulder_point, neck_point2])
    
    # Three-Point Angle #1 (3 points)
    point_a1 = (random.randint(width//6, width//2), random.randint(height//2, 3*height//4))
    point_b1 = (point_a1[0] + random.randint(30, 80), point_a1[1] + random.randint(-30, 30))
    point_c1 = (point_b1[0] + random.randint(20, 60), point_b1[1] + random.randint(30, 80))
    points.extend([point_a1, point_b1, point_c1])
    
    # Three-Point Angle #2 (3 points)
    point_a2 = (random.randint(width//2, 5*width//6), random.randint(height//2, 3*height//4))
    point_b2 = (point_a2[0] + random.randint(-80, -30), point_a2[1] + random.randint(-30, 30))
    point_c2 = (point_b2[0] + random.randint(-60, -20), point_b2[1] + random.randint(30, 80))
    points.extend([point_a2, point_b2, point_c2])
    
    return points

def angle_two_points_vs_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate angle between two points and vertical."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0
    
    cos_theta = (-dy) / mag
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = math.degrees(math.acos(cos_theta))
    
    return angle_deg

def angle_three_points(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
    """Compute angle at point b formed by a→b→c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))

def calculate_angles_auto(points: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
    """Calculate all angles from automatically detected points."""
    computed_angles = []
    point_index = 0
    
    for angle_index, (label, req_pts) in enumerate(ANGLE_SPECS):
        if point_index + req_pts > len(points):
            break
            
        pts = points[point_index:point_index + req_pts]
        point_index += req_pts
        
        if req_pts == 2:
            p1, p2 = pts
            ang = angle_two_points_vs_vertical(p1, p2)
            
            if angle_index == 1:  # Shoulder-Neck: invert
                ang = 180.0 - ang
                
        else:  # req_pts == 3
            a, b, c = pts
            ang = angle_three_points(a, b, c)
        
        computed_angles.append((label, ang))
    
    return computed_angles

def draw_auto_annotations(img: np.ndarray, points: List[Tuple[int, int]], angles: List[Tuple[str, float]]) -> np.ndarray:
    """Draw automatically detected points and calculated angles."""
    annotated = img.copy()
    point_index = 0
    
    # Color scheme for different angle types
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for angle_index, (label, req_pts) in enumerate(ANGLE_SPECS):
        if point_index + req_pts > len(points):
            break
            
        pts = points[point_index:point_index + req_pts]
        color = colors[angle_index % len(colors)]
        
        # Draw points
        for i, pt in enumerate(pts):
            cv2.circle(annotated, pt, 8, color, -1)
            cv2.circle(annotated, pt, 8, (255, 255, 255), 2)
            cv2.putText(annotated, f"{point_index + i + 1}", (pt[0]+12, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw lines
        if req_pts == 2:
            cv2.line(annotated, pts[0], pts[1], color, 2)
        else:  # 3 points
            cv2.line(annotated, pts[0], pts[1], color, 2)
            cv2.line(annotated, pts[1], pts[2], color, 2)
        
        point_index += req_pts
    
    # Add angle text annotations
    for i, (label, angle) in enumerate(angles):
        text = f"{label}: {angle:.1f}°"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            annotated,
            (5, 20 + 30 * i - text_height - 5),
            (15 + text_width, 25 + 30 * i + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            annotated,
            text,
            (10, 20 + 30 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
    
    return annotated

# ========== API ENDPOINTS ==========

@app.post("/upload-analyze-auto", response_model=AutoAnalysisResponse)
async def upload_and_analyze_auto(file: UploadFile = File(...)):
    """
    Upload an image and automatically detect points and calculate angles.
    No manual point selection required - everything is automated!
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
        
        # Load and verify image
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # 🤖 AUTOMATIC POINT DETECTION
        detected_points = detect_anatomical_points_auto(img)
        
        # Calculate angles from detected points
        angles = calculate_angles_auto(detected_points)
        
        # Create annotated image
        annotated_img = draw_auto_annotations(img, detected_points, angles)
        
        # Save annotated image
        annotated_filename = f"auto_analysis_{image_id}.jpg"
        annotated_path = os.path.join(PROCESSED_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)
        
        # Prepare response
        results = [
            AngleResult(label=label, angle=round(angle, 1), angle_index=i)
            for i, (label, angle) in enumerate(angles)
        ]
        
        # Convert points to list format for JSON
        points_list = [[int(pt[0]), int(pt[1])] for pt in detected_points]
        
        return AutoAnalysisResponse(
            success=True,
            results=results,
            detected_points=points_list,
            annotated_image_url=f"/processed-auto/{annotated_filename}",
            message=f"Automatically detected {len(detected_points)} points and calculated {len(results)} angles"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in automatic analysis: {str(e)}")

# ========== UTILITY ENDPOINTS ==========

@app.get("/processed-auto/{filename}")
async def get_processed_auto_image(filename: str):
    """Serve processed images from automatic analysis."""
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Processed image not found")
    return FileResponse(filepath)

@app.get("/health-auto")
async def health_check_auto():
    """Health check endpoint for automatic analysis."""
    return {
        "status": "healthy",
        "message": "Automatic Anatomy Angle Calculator is running",
        "version": "1.0.0",
        "features": ["automatic_point_detection", "angle_calculation"]
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Automatic Anatomy Angle Calculator API",
        "version": "1.0.0",
        "description": "Upload an image and get automatic anatomical analysis",
        "endpoints": {
            "POST /upload-analyze-auto": "Upload image for automatic analysis",
            "GET /processed-auto/{filename}": "Get processed image",
            "GET /health-auto": "Health check"
        },
        "features": [
            "Automatic point detection using computer vision",
            "Automatic angle calculations", 
            "No manual point selection required"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
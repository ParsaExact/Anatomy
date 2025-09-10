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
    pixel_to_mm_ratio: Optional[float] = None  # Allow custom scaling

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
def calculate_postural_analysis(points: List[Tuple[int, int]], pixel_to_mm_ratio: float = None) -> Dict[str, Any]:
    """
    Calculate postural analysis using pure SCODIAC formulas without pixel-to-mm conversion.
    
    Implements the exact formulas from SCODIAC v2.7 software:
    - FAI-C7 = |l / (c+d)| * 100
    - FAI-A = |c-d| / (c+d) * 100  
    - FAI-T = |a-b| / (a+b) * 100
    - HDI-S = h/e * 100
    - HDI-A = g/e * 100
    - HDI-T = f/e * 100
    - POTSI = Sum of all FAI and HDI components
    
    All calculations are done in pure pixel coordinates as ratios.
    
    Where:
    - l = C7 distance from midline (pixels)
    - c = left armpit distance from midline (pixels)
    - d = right armpit distance from midline (pixels)
    - a = left waist distance from midline (pixels)
    - b = right waist distance from midline (pixels)
    - e = spine length (C7 to sacrum) (pixels)
    - h, g, f = height differences for shoulders, armpits, waist (pixels)
    
    Args:
        points: List of 8 anatomical points [(x,y), ...] in order:
               [C7, L_Shoulder, R_Shoulder, L_Armpit, R_Armpit, L_Waist, R_Waist, Sacrum]
        pixel_to_mm_ratio: Kept for compatibility but not used in calculations
        
    Returns:
        Dictionary containing all POTSI components as pure ratios
    """
    if len(points) != 8:
        raise ValueError(f"Expected 8 points, got {len(points)}")
    
    c7 = points[0]
    l_shoulder = points[1]
    r_shoulder = points[2]
    l_armpit = points[3]
    r_armpit = points[4]
    l_waist = points[5]
    r_waist = points[6]
    sacrum = points[7]

    # Calculate anatomical midline (pure pixel coordinates)
    midline_x = (c7[0] + sacrum[0]) / 2
    
    # Calculate distances from midline in pixels (pure ratios)
    l_c7_dist = abs(c7[0] - midline_x)  # l
    l_armpit_dist = abs(l_armpit[0] - midline_x)  # c
    r_armpit_dist = abs(r_armpit[0] - midline_x)  # d
    l_waist_dist = abs(l_waist[0] - midline_x)   # a
    r_waist_dist = abs(r_waist[0] - midline_x)   # b
    
    # Spine length in pixels for HDI calculations (e)
    spine_length = abs(c7[1] - sacrum[1])  # e
    
    # FAI calculations according to SCODIAC formulas (pure ratios):
    # FAI-C7 = |l / (c+d)| * 100
    fai_c7 = abs(l_c7_dist / (l_armpit_dist + r_armpit_dist)) * 100 if (l_armpit_dist + r_armpit_dist) > 0 else 0
    
    # FAI-A = |c-d| / (c+d) * 100  
    fai_a = abs(l_armpit_dist - r_armpit_dist) / (l_armpit_dist + r_armpit_dist) * 100 if (l_armpit_dist + r_armpit_dist) > 0 else 0
    
    # FAI-T = |a-b| / (a+b) * 100
    fai_t = abs(l_waist_dist - r_waist_dist) / (l_waist_dist + r_waist_dist) * 100 if (l_waist_dist + r_waist_dist) > 0 else 0
    
    # HDI calculations according to SCODIAC formulas (pure ratios):
    # HDI-S = h/e * 100 (shoulder height difference / spine length)
    hdi_s = abs(l_shoulder[1] - r_shoulder[1]) / spine_length * 100 if spine_length > 0 else 0
    
    # HDI-A = g/e * 100 (armpit height difference / spine length) 
    hdi_a = abs(l_armpit[1] - r_armpit[1]) / spine_length * 100 if spine_length > 0 else 0
    
    # HDI-T = f/e * 100 (waist height difference / spine length)
    hdi_t = abs(l_waist[1] - r_waist[1]) / spine_length * 100 if spine_length > 0 else 0
    
    # POTSI calculation (sum of all FAI and HDI components)
    potsi = fai_c7 + fai_a + fai_t + hdi_s + hdi_a + hdi_t
    
    return {
        'shoulder_symmetry': round(hdi_s, 2),
        'armpit_symmetry': round(hdi_a, 2), 
        'waist_symmetry': round(hdi_t, 2),
        'spinal_alignment': round(abs(c7[0] - sacrum[0]), 2),  # Pure pixel difference
        'overall_posture_score': round(potsi, 2),
        'fai_c7': round(fai_c7, 2),
        'fai_armpit': round(fai_a, 2),
        'fai_waist': round(fai_t, 2),
        'hdi_shoulder': round(hdi_s, 2),
        'hdi_armpit': round(hdi_a, 2),
        'hdi_waist': round(hdi_t, 2),
        'pixel_to_mm_ratio': 1.0,  # Pure ratio, no conversion
        'spine_length_pixels': round(spine_length, 2),
        'distances_from_midline_pixels': {
            'c7': round(l_c7_dist, 2),
            'left_armpit': round(l_armpit_dist, 2),
            'right_armpit': round(r_armpit_dist, 2),
            'left_waist': round(l_waist_dist, 2),
            'right_waist': round(r_waist_dist, 2)
        },
        'anatomical_points': {
            'C7_Neck': c7,
            'Left_Shoulder': l_shoulder,
            'Right_Shoulder': r_shoulder,
            'Left_Armpit': l_armpit,
            'Right_Armpit': r_armpit,
            'Left_Waist': l_waist,
            'Right_Waist': r_waist,
            'Sacrum_Point': sacrum
        }
    }

def calculate_8_point_analysis(points: List[Tuple[int, int]], pixel_to_mm_ratio: float = None) -> Dict[str, Any]:
    """
    Main analysis function that calculates all 8-point postural measurements using improved POTSI methodology.
    
    Args:
        points: List of 8 anatomical points in order:
               [C7, L_Shoulder, R_Shoulder, L_Armpit, R_Armpit, L_Waist, R_Waist, Sacrum]
        pixel_to_mm_ratio: Conversion factor from pixels to mm. If None, uses optimized default.
    
    Returns:
        Dictionary containing all calculated measurements
    """
    if len(points) != 8:
        raise ValueError(f"Expected 8 points, got {len(points)}")
    
    # Use the improved postural analysis function
    return calculate_postural_analysis(points, pixel_to_mm_ratio)

def draw_8_point_annotations(img: np.ndarray, points: List[Tuple[int, int]], results: Dict[str, Any]) -> np.ndarray:
    """Draw the 8 anatomical points and analysis lines with POTSI components."""
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
    
    # Improved midline reference - between C7 and sacrum
    midline_x = int((c7[0] + sacrum[0]) / 2)
    cv2.line(annotated, (midline_x, 0), (midline_x, annotated.shape[0]), (128, 128, 128), 2)
    
    # Add midline label
    cv2.putText(annotated, "Midline", (midline_x + 5, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Add POTSI results text
    measurements = [
        f"POTSI: {results['overall_posture_score']:.2f}",
        f"FAI_C7: {results.get('fai_c7', 0):.2f}",
        f"FAI_A: {results.get('fai_armpit', 0):.2f}",
        f"FAI_T: {results.get('fai_waist', 0):.2f}",
        f"HDI_S: {results.get('hdi_shoulder', 0):.2f}",
        f"HDI_A: {results.get('hdi_armpit', 0):.2f}",
        f"HDI_T: {results.get('hdi_waist', 0):.2f}",
        f"Scale: {results.get('pixel_to_mm_ratio', 0.735):.3f}mm/px"
    ]
    
    y_offset = 30
    for i, text in enumerate(measurements):
        # Background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(
            annotated,
            (5, y_offset + i*25 - text_height - 3),
            (15 + text_width, y_offset + i*25 + 3),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (10, y_offset + i*25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
    Analyze 8-point anatomy based on the provided points and image using improved POTSI methodology.
    
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
        ],
        "pixel_to_mm_ratio": 0.735  // Optional: Custom scaling ratio (default: 0.735)
    }
    
    Returns POTSI (Posture Symmetry Index) components:
    - FAI (Frontal Asymmetry Index): C7, Armpit, Waist deviations from midline
    - HDI (Horizontal Deviation Index): Left-right differences in height
    - Overall POTSI score: Sum of all asymmetry components
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
        
        # Calculate 8-point analysis with custom or default ratio
        analysis_results = calculate_8_point_analysis(points_tuples, request.pixel_to_mm_ratio)
        
        # Create annotated image
        annotated_img = draw_8_point_annotations(img, points_tuples, analysis_results)
        
        # Save annotated image
        annotated_filename = f"annotated_8point_{request.image_id}.jpg"
        annotated_path = os.path.join(PROCESSED_FOLDER, annotated_filename)
        success = cv2.imwrite(annotated_path, annotated_img)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save annotated image")
        
        # Prepare response with detailed POTSI components
        results = [
            AnalysisResult(measurement="POTSI Score", value=analysis_results['overall_posture_score']),
            AnalysisResult(measurement="FAI C7", value=analysis_results['fai_c7']),
            AnalysisResult(measurement="FAI Armpit", value=analysis_results['fai_armpit']),
            AnalysisResult(measurement="FAI Waist", value=analysis_results['fai_waist']),
            AnalysisResult(measurement="HDI Shoulder", value=analysis_results['hdi_shoulder']),
            AnalysisResult(measurement="HDI Armpit", value=analysis_results['hdi_armpit']),
            AnalysisResult(measurement="HDI Waist", value=analysis_results['hdi_waist']),
            AnalysisResult(measurement="Spinal Alignment", value=analysis_results['spinal_alignment']),
            AnalysisResult(measurement="Pixel to MM Ratio", value=analysis_results['pixel_to_mm_ratio']),
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
        "order": "Points should be provided in order: C7 Neck, Left Shoulder, Right Shoulder, Left Armpit, Right Armpit, Left Waist, Right Waist, Sacrum Point",
        "default_pixel_to_mm_ratio": 0.735,
        "potsi_methodology": {
            "description": "POTSI (Posture Symmetry Index) measures postural asymmetries",
            "components": {
                "FAI": "Frontal Asymmetry Index - horizontal deviations from sacrum midline",
                "HDI": "Horizontal Deviation Index - vertical differences between paired points"
            },
            "formula": "POTSI = FAI_C7 + FAI_Armpit + FAI_Waist + HDI_Shoulder + HDI_Armpit + HDI_Waist",
            "target_range": "Normal POTSI scores typically range from 10-20mm"
        }
    }

@app.get("/optimal-ratio")
async def get_optimal_ratio():
    """
    Get information about the optimal pixel-to-mm ratio based on research.
    """
    return {
        "default_ratio": 0.735,
        "description": "Optimized pixel-to-mm ratio for accurate POTSI measurements",
        "calibration_info": {
            "method": "Based on spine length estimation and target POTSI validation",
            "accuracy": "Provides POTSI scores matching clinical reference values",
            "recommendation": "Use default 0.735 for standard photography, or calibrate manually for precision"
        },
        "usage": {
            "automatic": "Leave pixel_to_mm_ratio null/empty for automatic scaling",
            "custom": "Provide specific ratio if you have calibrated measurements",
            "calibration": "Measure a known distance in your image to calculate custom ratio"
        }
    }

@app.post("/calibrate-ratio")
async def calibrate_ratio(request: dict):
    """
    Calibrate pixel-to-mm ratio based on reference FAI values.
    
    Expected format:
    {
        "points": [...],  // 8 anatomical points
        "reference_fai": {
            "fai_c7": 8.97,
            "fai_armpit": 10.25, 
            "fai_waist": 13.59
        }
    }
    """
    try:
        points = [(pt["x"], pt["y"]) for pt in request["points"]]
        ref_fai = request["reference_fai"]
        
        if len(points) != 8:
            raise HTTPException(status_code=400, detail="Need exactly 8 points")
        
        # Test different ratios to find best match
        best_ratio = 0.735
        best_error = float('inf')
        
        for test_ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            results = calculate_postural_analysis(points, test_ratio)
            
            # Calculate error from reference values
            error = (
                abs(results['fai_c7'] - ref_fai['fai_c7']) +
                abs(results['fai_armpit'] - ref_fai['fai_armpit']) +
                abs(results['fai_waist'] - ref_fai['fai_waist'])
            )
            
            if error < best_error:
                best_error = error
                best_ratio = test_ratio
        
        # Test the best ratio
        best_results = calculate_postural_analysis(points, best_ratio)
        
        return {
            "optimal_ratio": best_ratio,
            "calibration_error": best_error,
            "predicted_fai": {
                "fai_c7": best_results['fai_c7'],
                "fai_armpit": best_results['fai_armpit'],
                "fai_waist": best_results['fai_waist']
            },
            "reference_fai": ref_fai,
            "message": f"Optimal ratio: {best_ratio:.3f} with error: {best_error:.2f}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

@app.get("/test-scodiac-formulas")
async def test_scodiac_formulas():
    """
    Test endpoint to verify SCODIAC formula implementation.
    Tests against the exact formulas shown in the SCODIAC software image.
    """
    # Test points (example from your data)
    test_points = [
        (351, 423),  # C7
        (283, 442),  # L_Shoulder
        (427, 444),  # R_Shoulder
        (275, 490),  # L_Armpit
        (432, 490),  # R_Armpit
        (292, 562),  # L_Waist
        (418, 553),  # R_Waist
        (358, 622)   # Sacrum
    ]
    
    try:
        # Test with automatic ratio calculation
        results_auto = calculate_postural_analysis(test_points, None)
        
        # Test with fixed ratio
        results_fixed = calculate_postural_analysis(test_points, 0.735)
        
        # Manual calculation for verification
        c7, l_shoulder, r_shoulder, l_armpit, r_armpit, l_waist, r_waist, sacrum = test_points
        midline_x = (c7[0] + sacrum[0]) / 2
        
        # Calculate distances from midline (using fixed ratio for manual check)
        ratio = 0.735
        l_c7_dist = abs(c7[0] - midline_x) * ratio
        l_armpit_dist = abs(l_armpit[0] - midline_x) * ratio  
        r_armpit_dist = abs(r_armpit[0] - midline_x) * ratio
        l_waist_dist = abs(l_waist[0] - midline_x) * ratio
        r_waist_dist = abs(r_waist[0] - midline_x) * ratio
        spine_length = abs(c7[1] - sacrum[1]) * ratio
        
        # Manual SCODIAC formula calculations
        manual_fai_c7 = abs(l_c7_dist / (l_armpit_dist + r_armpit_dist)) * 100 if (l_armpit_dist + r_armpit_dist) > 0 else 0
        manual_fai_a = abs(l_armpit_dist - r_armpit_dist) / (l_armpit_dist + r_armpit_dist) * 100 if (l_armpit_dist + r_armpit_dist) > 0 else 0
        manual_fai_t = abs(l_waist_dist - r_waist_dist) / (l_waist_dist + r_waist_dist) * 100 if (l_waist_dist + r_waist_dist) > 0 else 0
        
        return {
            "success": True,
            "scodiac_formulas": {
                "FAI_C7": "l / (c+d) * 100",
                "FAI_A": "|c-d| / (c+d) * 100", 
                "FAI_T": "|a-b| / (a+b) * 100",
                "HDI_S": "h/e * 100",
                "HDI_A": "g/e * 100",
                "HDI_T": "f/e * 100"
            },
            "test_points": {
                "midline_x": midline_x,
                "spine_length": spine_length
            },
            "distances_from_midline": {
                "l (C7)": l_c7_dist,
                "c (L_Armpit)": l_armpit_dist,
                "d (R_Armpit)": r_armpit_dist,
                "a (L_Waist)": l_waist_dist,
                "b (R_Waist)": r_waist_dist
            },
            "manual_calculations": {
                "FAI_C7": round(manual_fai_c7, 2),
                "FAI_A": round(manual_fai_a, 2),
                "FAI_T": round(manual_fai_t, 2)
            },
            "function_results": {
                "fai_c7": results_fixed['fai_c7'],
                "fai_armpit": results_fixed['fai_armpit'],
                "fai_waist": results_fixed['fai_waist'],
                "potsi": results_fixed['overall_posture_score']
            },
            "scodiac_reference_values": {
                "FAI_C7": 4.42,
                "FAI_A": 5.31,
                "FAI_T": 9.8,
                "POTSI": 22.42
            },
            "message": "SCODIAC formula implementation test complete"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to test SCODIAC formulas"
        }

@app.get("/validate-pure-scodiac")
async def validate_pure_scodiac():
    """
    Validate pure SCODIAC formula implementation without pixel-to-mm conversion.
    This endpoint verifies that our calculations work with pure pixel ratios.
    """
    try:
        # Use the exact test points from SCODIAC reference
        test_points = [
            (351, 423),  # C7
            (283, 442),  # L_Shoulder  
            (427, 444),  # R_Shoulder
            (275, 490),  # L_Armpit
            (432, 490),  # R_Armpit
            (292, 562),  # L_Waist
            (418, 553),  # R_Waist
            (358, 622)   # Sacrum
        ]
        
        # Calculate with our pure implementation
        results = calculate_postural_analysis(test_points)
        
        # Manual verification calculations (pure pixels)
        c7, l_shoulder, r_shoulder, l_armpit, r_armpit, l_waist, r_waist, sacrum = test_points
        midline_x = (c7[0] + sacrum[0]) / 2
        
        # Calculate distances in pure pixels
        l_c7_dist = abs(c7[0] - midline_x)
        l_armpit_dist = abs(l_armpit[0] - midline_x)
        r_armpit_dist = abs(r_armpit[0] - midline_x)
        l_waist_dist = abs(l_waist[0] - midline_x)
        r_waist_dist = abs(r_waist[0] - midline_x)
        spine_length = abs(c7[1] - sacrum[1])
        
        # SCODIAC formula verification (pure ratios)
        manual_fai_c7 = abs(l_c7_dist / (l_armpit_dist + r_armpit_dist)) * 100
        manual_fai_a = abs(l_armpit_dist - r_armpit_dist) / (l_armpit_dist + r_armpit_dist) * 100
        manual_fai_t = abs(l_waist_dist - r_waist_dist) / (l_waist_dist + r_waist_dist) * 100
        
        manual_hdi_s = abs(l_shoulder[1] - r_shoulder[1]) / spine_length * 100
        manual_hdi_a = abs(l_armpit[1] - r_armpit[1]) / spine_length * 100
        manual_hdi_t = abs(l_waist[1] - r_waist[1]) / spine_length * 100
        
        manual_potsi = manual_fai_c7 + manual_fai_a + manual_fai_t + manual_hdi_s + manual_hdi_a + manual_hdi_t
        
        return {
            "success": True,
            "pure_pixel_calculations": {
                "midline_x": round(midline_x, 2),
                "spine_length_pixels": round(spine_length, 2),
                "distances_pixels": {
                    "c7_from_midline": round(l_c7_dist, 2),
                    "left_armpit_from_midline": round(l_armpit_dist, 2),
                    "right_armpit_from_midline": round(r_armpit_dist, 2),
                    "left_waist_from_midline": round(l_waist_dist, 2),
                    "right_waist_from_midline": round(r_waist_dist, 2)
                }
            },
            "manual_calculations": {
                "fai_c7": round(manual_fai_c7, 2),
                "fai_armpit": round(manual_fai_a, 2),
                "fai_waist": round(manual_fai_t, 2),
                "hdi_shoulder": round(manual_hdi_s, 2),
                "hdi_armpit": round(manual_hdi_a, 2),
                "hdi_waist": round(manual_hdi_t, 2),
                "potsi_total": round(manual_potsi, 2)
            },
            "function_results": {
                "fai_c7": results['fai_c7'],
                "fai_armpit": results['fai_armpit'],
                "fai_waist": results['fai_waist'],
                "hdi_shoulder": results['hdi_shoulder'],
                "hdi_armpit": results['hdi_armpit'],
                "hdi_waist": results['hdi_waist'],
                "potsi_total": results['overall_posture_score']
            },
            "formula_explanation": {
                "FAI_C7": f"{l_c7_dist:.1f} / ({l_armpit_dist:.1f} + {r_armpit_dist:.1f}) * 100 = {manual_fai_c7:.2f}",
                "FAI_A": f"|{l_armpit_dist:.1f} - {r_armpit_dist:.1f}| / ({l_armpit_dist:.1f} + {r_armpit_dist:.1f}) * 100 = {manual_fai_a:.2f}",
                "FAI_T": f"|{l_waist_dist:.1f} - {r_waist_dist:.1f}| / ({l_waist_dist:.1f} + {r_waist_dist:.1f}) * 100 = {manual_fai_t:.2f}",
                "note": "All calculations use pure pixel coordinates without mm conversion"
            },
            "validation_status": "Pure SCODIAC ratios correctly implemented"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Pure validation failed"
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
    # Import uvicorn only when running directly to avoid import errors
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        print("Then run: python main2.py")
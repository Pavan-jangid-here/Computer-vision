from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DetectionResult:
    """Detection result container"""
    has_helmet: bool
    confidence: float
    person_bbox: Tuple[int, int, int, int]
    person_area: np.ndarray = None

class HelmetDetector:
    """YOLO-based helmet detection module"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        """
        Initialize helmet detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = {0:"Helmet", 1:"No Helmet", 2:"No Person"}
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect helmet status in image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of detection results
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Determine helmet status
                has_helmet = class_id == 0  # 0 = helmet, 1 = no_helmet
                
                detection = DetectionResult(
                    has_helmet=has_helmet,
                    confidence=confidence,
                    person_bbox=(x1, y1, x2, y2),
                    person_area=image[y1:y2, x1:x2].copy()
                )
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[DetectionResult]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.person_bbox
            color = (0, 255, 0) if det.has_helmet else (0, 0, 255)
            label = f"Helmet: {det.confidence:.2f}" \
                    if det.has_helmet \
                    else f"NO HELMET: {det.confidence:.2f}"
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output

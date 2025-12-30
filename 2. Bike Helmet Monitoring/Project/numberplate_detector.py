from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PlateDetectionResult:
    """Number plate detection result container"""
    plate_bbox: Tuple[int, int, int, int]
    confidence: float
    plate_area: np.ndarray = None


class NumberPlateDetector:
    """YOLO-based number plate detection module"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize number plate detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_name = "NumberPlate"
    
    def detect(self, image: np.ndarray) -> List[PlateDetectionResult]:
        """
        Detect number plates in image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of plate detection results
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extract plate region
                plate_area = image[y1:y2, x1:x2].copy()
                
                detection = PlateDetectionResult(
                    plate_bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    plate_area=plate_area
                )
                detections.append(detection)
        
        return detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[PlateDetectionResult]
    ) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.plate_bbox
            color = (255, 0, 0)  # Blue color for plates
            
            label = f"Plate {idx + 1}: {det.confidence:.2f}"
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]
            
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return output
    
    def get_plate_info(
        self,
        detection: PlateDetectionResult
    ) -> Dict:
        """
        Get detailed information about detected plate
        
        Args:
            detection: PlateDetectionResult object
            
        Returns:
            Dictionary with plate information
        """
        x1, y1, x2, y2 = detection.plate_bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        return {
            "bbox": (x1, y1, x2, y2),
            "confidence": detection.confidence,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": width / height if height > 0 else 0
        }

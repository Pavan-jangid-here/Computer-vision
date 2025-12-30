# ============================================================================
# HELMET DETECTION & NUMBER PLATE OCR SYSTEM
# ============================================================================
# Flow: Image Analysis → Helmet Detection → Number Plate Detection → 
#       OCR Processing → Database Storage
# ============================================================================

import os
import cv2
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import sqlite3
from dotenv import load_dotenv

from helmet_detector import HelmetDetector, DetectionResult
from numberplate_detector import NumberPlateDetector, PlateDetectionResult
from azure_ocr_processor import AzureOCRProcessor
from database_manager import DatabaseManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('helmet_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HelmetDetectionPipeline:
    """
    Complete pipeline for helmet detection and number plate recognition
    """
    
    def __init__(
        self,
        helmet_model_path: str = "Models/HelmetDetection.pt",
        plate_model_path: str = "Models/NumberPlateDetection.pt",
        db_path: str = "detection_results.db"
    ):
        """
        Initialize the detection pipeline
        
        Args:
            helmet_model_path: Path to helmet detection YOLO model
            plate_model_path: Path to number plate detection YOLO model
            db_path: Path to SQLite database
        """
        logger.info("Initializing Helmet Detection Pipeline...")
        
        # Initialize detectors
        self.helmet_detector = HelmetDetector(helmet_model_path, conf_threshold=0.25)
        self.plate_detector = NumberPlateDetector(plate_model_path, conf_threshold=0.25)
        
        # Initialize OCR processor
        self.ocr_processor = AzureOCRProcessor()
        
        # Initialize database
        self.db_manager = DatabaseManager(db_path)
        self.db_manager.create_tables()
        
        logger.info("Pipeline initialized successfully!")
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image through the entire pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing all detection and OCR results
        """
        logger.info(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {"status": "error", "message": "Image not found"}
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return {"status": "error", "message": "Failed to read image"}
        
        # Step 1: Helmet Detection
        logger.info("Step 1: Detecting helmet status...")
        helmet_detections = self.helmet_detector.detect(image)
        helmet_results = self._process_helmet_detections(helmet_detections)
        
        # Step 2: Number Plate Detection
        logger.info("Step 2: Detecting number plates...")
        plate_detections = self.plate_detector.detect(image)
        plate_results = self._process_plate_detections(image, plate_detections)
        
        # Step 3: OCR Processing
        logger.info("Step 3: Processing OCR for number plates...")
        ocr_results = self._process_ocr(plate_results, image)
        
        # Combine all results
        combined_results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "helmet_detection": helmet_results,
            "number_plate_detection": plate_results,
            "ocr_results": ocr_results,
            "status": "success"
        }
        
        # Step 4: Store in Database
        logger.info("Step 4: Storing results in database...")
        self.db_manager.store_detection(combined_results)
        
        return combined_results
    
    def _process_helmet_detections(self, detections: List[DetectionResult]) -> Dict:
        """
        Process helmet detection results
        """
        results = {
            "total_persons": len(detections),
            "with_helmet": 0,
            "without_helmet": 0,
            "detections": []
        }
        
        for det in detections:
            result_dict = {
                "has_helmet": det.has_helmet,
                "confidence": round(det.confidence, 4),
                "bbox": det.person_bbox,
                "status": "SAFE" if det.has_helmet else "VIOLATION"
            }
            results["detections"].append(result_dict)
            
            if det.has_helmet:
                results["with_helmet"] += 1
            else:
                results["without_helmet"] += 1
            
            logger.info(
                f"Person detected - Helmet: {det.has_helmet}, "
                f"Confidence: {det.confidence:.4f}"
            )
        
        return results
    
    def _process_plate_detections(
        self,
        image: np.ndarray,
        detections: List[PlateDetectionResult]
    ) -> Dict:
        """
        Process number plate detection results
        """
        results = {
            "total_plates": len(detections),
            "plates": []
        }
        
        for idx, det in enumerate(detections):
            plate_dict = {
                "plate_id": idx + 1,
                "bbox": det.plate_bbox,
                "confidence": round(det.confidence, 4),
                "plate_area": det.plate_area.shape if det.plate_area is not None else None,
                "status": "detected"
            }
            results["plates"].append(plate_dict)
            
            logger.info(
                f"Number plate {idx + 1} detected - "
                f"Confidence: {det.confidence:.4f}"
            )
        
        return results
    
    def _process_ocr(self, plate_results: Dict, image: np.ndarray) -> Dict:
        """
        Process OCR for detected number plates
        """
        ocr_results = {
            "processed_plates": 0,
            "extraction_results": []
        }
        
        # Extract plate areas and process with OCR
        plate_detections = self.plate_detector.detect(image)
        
        for idx, det in enumerate(plate_detections):
            if det.plate_area is None:
                continue
            
            # Extract text using Azure OCR
            ocr_text = self.ocr_processor.extract_text_from_array(det.plate_area)
            
            if ocr_text:
                extraction = {
                    "plate_id": idx + 1,
                    "extracted_text": ocr_text,
                    "confidence": "high" if len(ocr_text) > 0 else "low",
                    "valid_format": self._validate_plate_format(ocr_text)
                }
                ocr_results["extraction_results"].append(extraction)
                ocr_results["processed_plates"] += 1
                
                logger.info(f"OCR Result - Plate {idx + 1}: {ocr_text}")
            else:
                logger.warning(f"OCR failed for plate {idx + 1}")
        
        return ocr_results
    
    @staticmethod
    def _validate_plate_format(plate_text: str) -> bool:
        """
        Validate if extracted text matches number plate format
        Indian format: 2 letters + 2 digits + 4 alphanumeric
        """
        plate_text = plate_text.strip().upper()
        # Simple validation - can be enhanced based on region
        return len(plate_text) >= 8 and any(c.isdigit() for c in plate_text)
    
    def process_batch(self, image_dir: str, output_file: str = "batch_results.json"):
        """
        Process multiple images from a directory
        
        Args:
            image_dir: Directory containing images
            output_file: Path to save batch results
        """
        logger.info(f"Processing batch from directory: {image_dir}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = {
            "batch_info": {
                "processed_date": datetime.now().isoformat(),
                "total_images": len(image_files),
                "output_file": output_file
            },
            "results": []
        }
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            logger.info(f"Processing [{idx}/{len(image_files)}]: {image_file}")
            
            result = self.process_image(image_path)
            results["results"].append(result)
        
        # Save results to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch processing complete. Results saved to {output_file}")
        return results
    
    def visualize_results(
        self,
        image_path: str,
        save_path: str = "output_visualization.jpg"
    ) -> str:
        """
        Create visualization of detection results
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        logger.info(f"Creating visualization for: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Failed to read image")
            return None
        
        # Get detections
        helmet_dets = self.helmet_detector.detect(image)
        plate_dets = self.plate_detector.detect(image)
        
        # Draw helmet detections
        output = self.helmet_detector.draw_detections(image.copy(), helmet_dets)
        
        # Draw plate detections
        output = self.plate_detector.draw_detections(output, plate_dets)
        
        # Save visualization
        cv2.imwrite(save_path, output)
        logger.info(f"Visualization saved to: {save_path}")
        
        return save_path


def main():
    """
    Main execution function
    """
    logger.info("=" * 80)
    logger.info("HELMET DETECTION & NUMBER PLATE OCR SYSTEM")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = HelmetDetectionPipeline(
        helmet_model_path="Models/HelmetDetection.pt",
        plate_model_path="Models/NumberPlateDetection.pt",
        db_path="detection_results.db"
    )
    
    # Example: Process single image
    test_image = "test4.png"  # Replace with actual image
    
    if os.path.exists(test_image):
        result = pipeline.process_image(test_image)
        logger.info(f"Processing result: {result}")
        
        # Create visualization
        pipeline.visualize_results(test_image, "visualization_output.jpg")
    else:
        logger.warning(f"Test image not found: {test_image}")
    
    # Example: Batch processing
    # pipeline.process_batch("images_directory/", "batch_results.json")


if __name__ == "__main__":
    main()

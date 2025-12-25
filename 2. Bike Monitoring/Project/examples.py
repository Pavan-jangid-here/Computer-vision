"""
Example usage scripts for the Helmet Detection & Number Plate OCR System
Demonstrates various ways to use the pipeline
"""

import os
import json
import logging
from pathlib import Path
from main import HelmetDetectionPipeline
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_single_image_processing():
    """
    Example 1: Process a single image and view results
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Single Image Processing")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = HelmetDetectionPipeline(
        helmet_model_path="Models/HelmetDetection.pt",
        plate_model_path="Models/NumberPlateDetection.pt",
        db_path="detection_results.db"
    )
    
    # Process image
    image_path = "test4.png"
    
    if os.path.exists(image_path):
        result = pipeline.process_image(image_path)
        
        # Print results
        print("\n" + "="*80)
        print("DETECTION RESULTS")
        print("="*80)
        print(json.dumps(result, indent=2))
        
    else:
        logger.warning(f"Image not found: {image_path}")


def example_2_batch_processing():
    """
    Example 2: Process multiple images from a directory
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Batch Processing")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = HelmetDetectionPipeline()
    
    # Process batch
    image_dir = "images/"
    
    if os.path.exists(image_dir):
        results = pipeline.process_batch(
            image_dir=image_dir,
            output_file="batch_results.json"
        )
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total Images Processed: {results['batch_info']['total_images']}")
        print(f"Output File: {results['batch_info']['output_file']}")
        
    else:
        logger.warning(f"Directory not found: {image_dir}")


def example_3_violation_analysis():
    """
    Example 3: Query and analyze violations from database
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Violation Analysis")
    logger.info("=" * 80)
    
    db = DatabaseManager("detection_results.db")
    
    # Get statistics
    stats = db.get_statistics()
    
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    print(f"Total Detections: {stats.get('total_detections', 0)}")
    print(f"Total Violations: {stats.get('total_violations', 0)}")
    print(f"Helmet Violations: {stats.get('helmet_violations', 0)}")
    print(f"Total OCR Results: {stats.get('total_ocr_results', 0)}")
    print(f"Valid OCR Results: {stats.get('valid_ocr_results', 0)}")
    print(f"OCR Success Rate: {stats.get('ocr_success_rate', 0):.2f}%")
    
    # Get recent violations
    violations = db.get_violations(limit=10, days=7)
    
    if violations:
        print("\n" + "="*80)
        print("RECENT VIOLATIONS (Last 7 Days)")
        print("="*80)
        for violation in violations:
            print(f"ID: {violation['id']}")
            print(f"  Type: {violation['violation_type']}")
            print(f"  Plate: {violation['number_plate_text']}")
            print(f"  Time: {violation['violation_timestamp']}")
            print()
    
    db.close()


def example_4_export_violations():
    """
    Example 4: Export violations to CSV for reporting
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Export Violations to CSV")
    logger.info("=" * 80)
    
    db = DatabaseManager("detection_results.db")
    
    # Export to CSV
    output_file = "violations_report.csv"
    db.export_violations_csv(output_file)
    
    if os.path.exists(output_file):
        print(f"\n✓ Violations exported to: {output_file}")
        
        # Show summary
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"✓ Total records exported: {len(lines) - 1}")
    
    db.close()


def example_5_visualization():
    """
    Example 5: Generate visualization of detection results
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Generate Visualization")
    logger.info("=" * 80)
    
    pipeline = HelmetDetectionPipeline()
    
    image_path = "test_image.jpg"
    output_path = "visualization_output.jpg"
    
    if os.path.exists(image_path):
        result = pipeline.visualize_results(
            image_path=image_path,
            save_path=output_path
        )
        
        if result:
            print(f"\n✓ Visualization saved to: {result}")
        else:
            logger.error("Failed to create visualization")
    else:
        logger.warning(f"Image not found: {image_path}")


def example_6_direct_detector_usage():
    """
    Example 6: Use detectors directly without pipeline
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Direct Detector Usage")
    logger.info("=" * 80)
    
    import cv2
    from helmet_detector import HelmetDetector
    from numberplate_detector import NumberPlateDetector
    
    # Load image
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    # Helmet detection
    helmet_detector = HelmetDetector("Models/HelmetDetection.pt")
    helmet_detections = helmet_detector.detect(image)
    
    print("\n" + "="*80)
    print("HELMET DETECTIONS")
    print("="*80)
    print(f"Total persons detected: {len(helmet_detections)}")
    for idx, det in enumerate(helmet_detections, 1):
        print(f"  Person {idx}:")
        print(f"    Has Helmet: {det.has_helmet}")
        print(f"    Confidence: {det.confidence:.4f}")
        print(f"    BBox: {det.person_bbox}")
    
    # Number plate detection
    plate_detector = NumberPlateDetector("Models/NumberPlateDetection.pt")
    plate_detections = plate_detector.detect(image)
    
    print("\n" + "="*80)
    print("NUMBER PLATE DETECTIONS")
    print("="*80)
    print(f"Total plates detected: {len(plate_detections)}")
    for idx, det in enumerate(plate_detections, 1):
        print(f"  Plate {idx}:")
        print(f"    Confidence: {det.confidence:.4f}")
        print(f"    BBox: {det.plate_bbox}")
        
        # Get plate info
        info = plate_detector.get_plate_info(det)
        print(f"    Dimensions: {info['width']}x{info['height']}")
        print(f"    Area: {info['area']}")


def example_7_custom_processing():
    """
    Example 7: Custom processing workflow
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 7: Custom Processing Workflow")
    logger.info("=" * 80)
    
    from helmet_detector import HelmetDetector
    from numberplate_detector import NumberPlateDetector
    from azure_ocr_processor import AzureOCRProcessor
    import cv2
    
    image_path = "test_image.jpg"
    
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    # Step 1: Detect helmets
    print("\nStep 1: Helmet Detection...")
    helmet_detector = HelmetDetector("Models/HelmetDetection.pt")
    helmet_dets = helmet_detector.detect(image)
    
    violations = []
    for det in helmet_dets:
        if not det.has_helmet:
            violations.append({
                "type": "NO_HELMET",
                "confidence": det.confidence
            })
    
    if violations:
        print(f"✗ Found {len(violations)} helmet violation(s)")
    else:
        print("✓ All persons wearing helmets")
    
    # Step 2: Detect number plates
    print("\nStep 2: Number Plate Detection...")
    plate_detector = NumberPlateDetector("Models/NumberPlateDetection.pt")
    plate_dets = plate_detector.detect(image)
    
    if plate_dets:
        print(f"✓ Detected {len(plate_dets)} number plate(s)")
        
        # Step 3: OCR
        print("\nStep 3: OCR Processing...")
        try:
            ocr = AzureOCRProcessor()
            
            for idx, plate_det in enumerate(plate_dets, 1):
                if plate_det.plate_area is not None:
                    result = ocr.extract_and_format_plate(plate_det.plate_area)
                    print(f"  Plate {idx}: {result['cleaned_text']}")
        
        except ValueError as e:
            print(f"⚠ OCR not available: {e}")
    else:
        print("✗ No number plates detected")


def example_8_database_stats():
    """
    Example 8: Comprehensive database statistics
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 8: Database Statistics")
    logger.info("=" * 80)
    
    db = DatabaseManager("detection_results.db")
    
    stats = db.get_statistics()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS")
    print("="*80)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.2f}%")
        else:
            print(f"{key:.<40} {value}")
    
    db.close()


def run_all_examples():
    """
    Run all examples (modify as needed)
    """
    print("\n" + "="*80)
    print("HELMET DETECTION & NUMBER PLATE OCR SYSTEM")
    print("EXAMPLE USAGE DEMONSTRATIONS")
    print("="*80 + "\n")
    
    # Run examples
    try:
        # example_1_single_image_processing()
        # example_2_batch_processing()
        example_3_violation_analysis()
        # example_4_export_violations()
        # example_5_visualization()
        # example_6_direct_detector_usage()
        # example_7_custom_processing()
        example_8_database_stats()
    
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # example_1_single_image_processing()
    # example_2_batch_processing()
    # example_3_violation_analysis()
    # example_4_export_violations()
    # example_5_visualization()
    # example_6_direct_detector_usage()
    # example_7_custom_processing()
    # example_8_database_stats()
    
    # Or run all
    run_all_examples()

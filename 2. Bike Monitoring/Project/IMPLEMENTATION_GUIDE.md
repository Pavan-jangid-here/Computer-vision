# Project Structure & Implementation Guide

## Directory Layout

```
helmet-detection-system/
├── main.py                           # Main pipeline orchestrator
├── helmet_detector.py                # Helmet detection module (from attachment)
├── numberplate_detector.py           # Number plate detection module
├── azure_ocr_processor.py            # Azure OCR integration
├── database_manager.py               # SQLite database management
├── config.py                         # Configuration constants
├── examples.py                       # Usage examples
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── README.md                         # Documentation
├── Models/
│   ├── HelmetDetection.pt           # Your helmet detection model
│   └── NumberPlateDetection.pt       # Your number plate detection model
├── output/                          # Output images and visualizations
├── logs/                            # Application logs
└── detection_results.db             # SQLite database (auto-created)
```

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT IMAGE                                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  HELMET DETECTION (YOLO)   │
                    │  - Detect persons         │
                    │  - Check helmet status    │
                    │  - Confidence scores      │
                    └─────────────┬──────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │  RECORD HELMET DETECTION RESULTS   │
                │  - Person count                    │
                │  - Helmet vs No Helmet             │
                │  - Bounding boxes                  │
                │  - Confidence scores               │
                └─────────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  NUMBER PLATE DETECTION (YOLO) │
                    │  - Detect plates               │
                    │  - Extract plate region        │
                    │  - Calculate confidence        │
                    └─────────────┬──────────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │  RECORD PLATE DETECTION RESULTS    │
                │  - Plate count                     │
                │  - Bounding boxes                  │
                │  - Plate dimensions                │
                │  - Confidence scores               │
                └─────────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  AZURE OCR PROCESSING          │
                    │  - Send to Azure API           │
                    │  - Extract text from plates    │
                    │  - Validate format             │
                    └─────────────┬──────────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │  RECORD OCR RESULTS                │
                │  - Extracted text                  │
                │  - Cleaned text                    │
                │  - Validation status               │
                │  - Confidence level                │
                └─────────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  VIOLATION ANALYSIS             │
                    │  - No helmet violations        │
                    │  - Match with plates           │
                    │  - Severity classification     │
                    └─────────────┬──────────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │  DATABASE STORAGE                  │
                │  - detection_records               │
                │  - helmet_detections               │
                │  - number_plate_detections         │
                │  - ocr_results                     │
                │  - violations                      │
                └─────────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  VISUALIZATION & REPORTS       │
                    │  - Annotated images            │
                    │  - Violation reports           │
                    │  - Statistics                  │
                    │  - CSV exports                 │
                    └────────────────────────────────┘
```

## Module Descriptions

### 1. helmet_detector.py (From Your Attachment)
**Purpose**: Detect whether persons are wearing helmets

**Key Classes**:
- `DetectionResult`: Data class for helmet detection results
- `HelmetDetector`: Main detector class

**Key Methods**:
- `__init__(model_path, conf_threshold)`: Initialize with YOLO model
- `detect(image)`: Detect helmets in image
- `draw_detections(image, detections)`: Draw bounding boxes

**Output**: List of `DetectionResult` objects with helmet status

### 2. numberplate_detector.py (Created)
**Purpose**: Detect number plates in images

**Key Classes**:
- `PlateDetectionResult`: Data class for plate detection
- `NumberPlateDetector`: Main detector class

**Key Methods**:
- `detect(image)`: Detect number plates
- `draw_detections(image, detections)`: Visualize detections
- `get_plate_info(detection)`: Extract plate metadata

**Output**: List of `PlateDetectionResult` objects with plate regions

### 3. azure_ocr_processor.py (Enhanced)
**Purpose**: Extract text from number plates using Azure AI

**Key Classes**:
- `AzureOCRProcessor`: Azure integration handler

**Key Methods**:
- `extract_text_from_file(image_path)`: OCR from file
- `extract_text_from_array(image_array)`: OCR from numpy array
- `extract_and_format_plate(plate_array)`: OCR with formatting
- `_validate_plate_text(text)`: Validate plate format

**Dependencies**: Azure Computer Vision API

### 4. database_manager.py (Created)
**Purpose**: Persistent storage of detection results

**Key Classes**:
- `DatabaseManager`: SQLite database handler

**Key Methods**:
- `create_tables()`: Initialize database schema
- `store_detection(result)`: Store complete detection result
- `get_violations(limit, days)`: Query violations
- `get_statistics()`: System statistics
- `export_violations_csv(output_file)`: Generate reports

**Database Tables**:
```
detection_records
  ├── id (PK)
  ├── timestamp
  ├── image_path
  └── processing_status

helmet_detections
  ├── id (PK)
  ├── detection_id (FK)
  ├── has_helmet
  ├── confidence
  └── bbox_x1, y1, x2, y2

number_plate_detections
  ├── id (PK)
  ├── detection_id (FK)
  ├── confidence
  └── bbox_x1, y1, x2, y2

ocr_results
  ├── id (PK)
  ├── plate_detection_id (FK)
  ├── extracted_text
  ├── cleaned_text
  └── valid_format

violations
  ├── id (PK)
  ├── detection_id (FK)
  ├── violation_type
  ├── severity
  ├── number_plate_text
  └── violation_timestamp
```

### 5. main.py (Created)
**Purpose**: Orchestrate complete pipeline

**Key Classes**:
- `HelmetDetectionPipeline`: Main pipeline orchestrator

**Key Methods**:
- `process_image(image_path)`: Process single image
- `process_batch(image_dir, output_file)`: Process multiple images
- `visualize_results(image_path, save_path)`: Generate visualization

**Workflow**:
1. Initialize detectors and database
2. Load image
3. Run helmet detection
4. Run plate detection
5. Run OCR on plates
6. Analyze for violations
7. Store in database
8. Return results

### 6. config.py (Created)
**Purpose**: Centralized configuration management

**Contains**:
- Model paths
- Database paths
- Confidence thresholds
- Logging configuration
- Processing parameters
- Feature flags

## Implementation Notes

### For Your Current Setup

1. **Helmet Detection Model**
   - Your model files remain in `Models/` folder
   - Classes: {0:"Helmet", 1:"No Helmet", 2:"No Person"}
   - Already has proper detection logic

2. **Number Plate Detection Model**
   - Similar YOLO architecture
   - Single class: "NumberPlate"
   - Provides bounding box for plate region

3. **Azure OCR**
   - Requires Azure Computer Vision API key
   - Set in .env file as AZURE_KEY and AZURE_ENDPOINT
   - Processes plate images for text extraction

4. **Database**
   - SQLite for local storage
   - Auto-created on first run
   - Scales well for medium-volume deployments

## Key Features Implemented

### 1. Complete Pipeline
✅ Helmet detection → Plate detection → OCR → Database storage

### 2. Error Handling
✅ Graceful handling of missing images
✅ Azure API error recovery
✅ Database transaction management

### 3. Logging
✅ Detailed logging to file and console
✅ Error tracking and debugging
✅ Performance monitoring

### 4. Data Persistence
✅ SQLite database for all results
✅ Historical tracking
✅ Violation records
✅ Statistics generation

### 5. Batch Processing
✅ Process multiple images
✅ JSON output for results
✅ Parallel processing ready

### 6. Visualization
✅ Draw detection boxes
✅ Label with confidence scores
✅ Save annotated images

### 7. Reporting
✅ CSV export for violations
✅ Statistics queries
✅ Date-range filtering

## Security Considerations

1. **API Keys**: Keep .env file with credentials (don't commit)
2. **Database Access**: SQLite not suitable for concurrent access
3. **Image Storage**: Implement proper access controls for saved images
4. **Data Privacy**: Consider GDPR/privacy for vehicle plate data

## Performance Optimization Tips

1. **Batch Size**: Process 10-20 images in parallel
2. **Model Quantization**: Use int8 models for faster inference
3. **Image Preprocessing**: Resize large images before processing
4. **Database Indexing**: Already indexed on common queries
5. **Caching**: Cache repeated plate OCR results

## Deployment Scenarios

### Local Development
```
python main.py
```

### Server-based Processing
```
python main.py --batch-mode --input-dir /data/images/
```

### API Integration
```
from main import HelmetDetectionPipeline
pipeline = HelmetDetectionPipeline()
result = pipeline.process_image(image_path)
```

## Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| Model not found | Verify Models/ directory and file names |
| Azure auth failed | Check AZURE_KEY and AZURE_ENDPOINT in .env |
| Database locked | Close other connections to database |
| OCR timeout | Increase MAX_RETRIES in config.py |
| Low helmet accuracy | Retrain with more diverse data |
| Plate OCR fails | Check plate image quality, try preprocessing |

## Future Enhancements

1. **Real-time Video Processing**: Extend to video streams
2. **REST API**: Wrap pipeline in Flask/FastAPI
3. **Multi-GPU Support**: Batch processing with CUDA
4. **Advanced Analytics**: Temporal analysis of violations
5. **ML Model Improvements**: Fine-tune models for your use case
6. **Cloud Integration**: Azure Blob Storage for images
7. **Notification System**: Alert on violations
8. **Web Dashboard**: Visualize statistics and results

## Contact & Support

For issues specific to your implementation, consider:
- Your helmet detection model's original documentation
- Azure Computer Vision SDK documentation
- YOLO (Ultralytics) documentation

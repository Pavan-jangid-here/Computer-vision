# ============================================================================
# HELMET DETECTION & NUMBER PLATE OCR SYSTEM - CONFIGURATION
# ============================================================================
# Environment configuration and constants
# ============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL PATHS
# ============================================================================
HELMET_MODEL_PATH = MODELS_DIR / "HelmetDetection.pt"
PLATE_MODEL_PATH = MODELS_DIR / "NumberPlateDetection.pt"

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_PATH = PROJECT_ROOT / "detection_results.db"

# ============================================================================
# AZURE CREDENTIALS
# ============================================================================
AZURE_KEY = os.getenv('AZURE_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================
HELMET_CONFIDENCE_THRESHOLD = float(os.getenv('HELMET_CONF_THRESHOLD', '0.5'))
PLATE_CONFIDENCE_THRESHOLD = float(os.getenv('PLATE_CONF_THRESHOLD', '0.4'))

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = LOG_DIR / "helmet_detection.log"

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MAX_IMAGE_SIZE = 2048  # Maximum dimension for resizing
RESIZE_QUALITY = 85    # JPEG quality for output

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================
BATCH_SIZE = 10           # Number of images to process in parallel
MAX_RETRIES = 30          # Max retries for Azure OCR
OCR_WAIT_TIME = 1.0       # Wait time between OCR polls (seconds)

# ============================================================================
# VIOLATION SETTINGS
# ============================================================================
VIOLATION_SEVERITY_HIGH = "HIGH"      # Not wearing helmet
VIOLATION_SEVERITY_LOW = "LOW"        # Other violations

# ============================================================================
# PLATE VALIDATION
# ============================================================================
MIN_PLATE_WIDTH = 50
MIN_PLATE_HEIGHT = 30
MIN_PLATE_ASPECT_RATIO = 2.0  # Width/Height ratio

# ============================================================================
# FEATURE FLAGS
# ============================================================================
ENABLE_VISUALIZATION = True      # Draw bounding boxes on output
ENABLE_DATABASE_STORAGE = True   # Store results in database
ENABLE_CSV_EXPORT = True         # Export violations to CSV
SAVE_CROPPED_PLATES = True       # Save cropped plate images

# ============================================================================
# API TIMEOUT SETTINGS
# ============================================================================
AZURE_API_TIMEOUT = 60  # seconds

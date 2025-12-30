"""
FIXED Azure OCR Processor - Handles both file and array inputs correctly
Fixes: 'bytes' object has no attribute 'read' error
"""

import os
import logging
import cv2
import numpy as np
import time
from typing import Optional
from io import BytesIO
from dotenv import load_dotenv

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes

load_dotenv()
logger = logging.getLogger(__name__)


class AzureOCRProcessor:
    """
    FIXED Azure Computer Vision OCR processor for text extraction
    """
    
    def __init__(self):
        """
        Initialize Azure OCR processor
        Requires AZURE_KEY and AZURE_ENDPOINT in environment variables
        """
        self.api_key = os.getenv('AZURE_KEY')
        self.endpoint = os.getenv('AZURE_ENDPOINT')
        
        if not self.api_key or not self.endpoint:
            logger.error(
                "Azure credentials not found. "
                "Set AZURE_KEY and AZURE_ENDPOINT in .env file"
            )
            raise ValueError("Azure credentials not configured")
        
        try:
            self.client = ComputerVisionClient(
                endpoint=self.endpoint.rstrip('/'),
                credentials=CognitiveServicesCredentials(self.api_key)
            )
            logger.info("Azure Computer Vision client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {e}")
            raise
    
    def extract_text_from_file(self, image_path: str) -> Optional[str]:
        """
        Extract text from image file using Azure OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text or None if failed
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            logger.info(f"Extracting text from file: {image_path}")
            
            with open(image_path, 'rb') as img_file:
                response = self.client.read_in_stream(
                    img_file,
                    raw=True
                )
            
            operation_location = response.headers['Operation-Location']
            operation_id = operation_location.split('/')[-1]
            
            logger.info(f"OCR operation started: {operation_id}")
            
            extracted_text = self._poll_for_results(operation_id)
            return extracted_text
        
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return None
    
    def extract_text_from_array(self, image_array: np.ndarray) -> Optional[str]:
        """
        FIXED: Extract text from numpy array (image)
        
        Args:
            image_array: Input image as numpy array (BGR format)
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info("Extracting text from image array...")
            
            # CRITICAL FIX: Encode as JPEG and create BytesIO object
            # Don't pass raw bytes - Azure API needs a file-like object
            _, img_encoded = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Create BytesIO object (file-like object)
            img_bytes_io = BytesIO(img_encoded.tobytes())
            
            logger.info("Sending image array to Azure OCR...")
            
            # Send to Azure using file-like object
            response = self.client.read_in_stream(
                img_bytes_io,
                raw=True
            )
            
            operation_location = response.headers['Operation-Location']
            operation_id = operation_location.split('/')[-1]
            
            logger.info(f"OCR operation started: {operation_id}")
            
            extracted_text = self._poll_for_results(operation_id)
            return extracted_text
        
        except Exception as e:
            logger.error(f"Error processing image array: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _poll_for_results(
        self,
        operation_id: str,
        max_retries: int = 30,
        wait_time: float = 1.0
    ) -> Optional[str]:
        """
        Poll Azure for OCR results
        
        Args:
            operation_id: Operation ID from Azure
            max_retries: Maximum number of retries
            wait_time: Wait time between retries (seconds)
            
        Returns:
            Extracted text or None
        """
        for attempt in range(max_retries):
            try:
                result = self.client.get_read_result(operation_id)
                
                if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                    if result.status == OperationStatusCodes.succeeded:
                        return self._extract_from_result(result)
                    else:
                        logger.warning(f"OCR failed. Status: {result.status}")
                        return None
                
                logger.info(f"Polling attempt {attempt + 1}/{max_retries}. Status: {result.status}")
                time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Error polling results: {e}")
                return None
        
        logger.warning("OCR polling timeout")
        return None
    
    @staticmethod
    def _extract_from_result(result) -> Optional[str]:
        """
        Extract text from Azure result object
        
        Args:
            result: Azure OCR result
            
        Returns:
            Concatenated extracted text
        """
        extracted_text = []
        
        try:
            if result.analyze_result and result.analyze_result.read_results:
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        extracted_text.append(line.text)
            
            if extracted_text:
                text = " ".join(extracted_text)
                logger.info(f"Text extracted successfully: {text[:100]}...")
                return text
            else:
                logger.warning("No text found in OCR result")
                return None
        
        except Exception as e:
            logger.error(f"Error extracting text from result: {e}")
            return None
    
    def extract_and_format_plate(
        self,
        plate_array: np.ndarray
    ) -> dict:
        """
        Extract and format number plate text
        
        Args:
            plate_array: Cropped number plate image
            
        Returns:
            Dictionary with extracted text and metadata
        """
        extracted_text = self.extract_text_from_array(plate_array)
        
        if not extracted_text:
            return {
                "text": None,
                "cleaned_text": None,
                "valid": False,
                "confidence": "low"
            }
        
        cleaned_text = self._clean_plate_text(extracted_text)
        is_valid = self._validate_plate_text(cleaned_text)
        
        return {
            "text": extracted_text,
            "cleaned_text": cleaned_text,
            "valid": is_valid,
            "confidence": "high" if is_valid else "medium"
        }
    
    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """
        Clean extracted number plate text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        cleaned = text.strip().upper()
        cleaned = ''.join(c for c in cleaned if c.isalnum())
        return cleaned
    
    @staticmethod
    def _validate_plate_text(text: str) -> bool:
        """
        Validate if text matches number plate format
        
        Args:
            text: Cleaned number plate text
            
        Returns:
            True if valid format
        """
        if not text:
            return False
        
        if len(text) < 8:
            return False
        
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        
        return has_letters and has_digits


def test_ocr_processor():
    """
    Test OCR processor functionality
    """
    try:
        logger.info("Initializing OCR processor for testing...")
        processor = AzureOCRProcessor()
        
        test_image = "test_plate.jpg"
        if os.path.exists(test_image):
            logger.info(f"Testing with image: {test_image}")
            result = processor.extract_and_format_plate(cv2.imread(test_image))
            logger.info(f"OCR Result: {result}")
            return result
        else:
            logger.warning(f"Test image not found: {test_image}")
            return None
    
    except Exception as e:
        logger.error(f"OCR processor test failed: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_ocr_processor()

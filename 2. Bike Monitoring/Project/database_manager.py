"""
Database Manager for storing detection results
SQLite database for helmet detection and number plate OCR results
"""

import sqlite3
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for detection results
    """
    
    def __init__(self, db_path: str = "detection_results.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Detection records table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Helmet detection table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS helmet_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    person_id INTEGER,
                    has_helmet BOOLEAN,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    status TEXT,
                    FOREIGN KEY (detection_id) REFERENCES detection_records(id)
                )
            ''')
            
            # Number plate detection table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS number_plate_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    plate_id INTEGER,
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    status TEXT,
                    FOREIGN KEY (detection_id) REFERENCES detection_records(id)
                )
            ''')
            
            # OCR results table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_detection_id INTEGER NOT NULL,
                    extracted_text TEXT,
                    cleaned_text TEXT,
                    confidence TEXT,
                    valid_format BOOLEAN,
                    processing_time REAL,
                    FOREIGN KEY (plate_detection_id) REFERENCES number_plate_detections(id)
                )
            ''')
            
            # Violation records table (for non-helmet wearers)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    helmet_detection_id INTEGER,
                    number_plate_text TEXT,
                    violation_type TEXT,
                    severity TEXT,
                    image_path TEXT,
                    violation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (detection_id) REFERENCES detection_records(id),
                    FOREIGN KEY (helmet_detection_id) REFERENCES helmet_detections(id)
                )
            ''')
            
            # Create indexes for faster queries
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_detection_timestamp 
                ON detection_records(timestamp)
            ''')
            
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                ON violations(violation_timestamp)
            ''')
            
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_helmet_status 
                ON helmet_detections(status)
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def store_detection(self, result: Dict) -> int:
        """
        Store complete detection result in database
        
        Args:
            result: Dictionary containing all detection and OCR results
            
        Returns:
            Detection record ID
        """
        try:
            # Insert main detection record
            self.cursor.execute('''
                INSERT INTO detection_records 
                (timestamp, image_path, processing_status)
                VALUES (?, ?, ?)
            ''', (
                result.get('timestamp'),
                result.get('image_path'),
                result.get('status', 'processing')
            ))
            
            detection_id = self.cursor.lastrowid
            
            # Store helmet detections
            helmet_data = result.get('helmet_detection', {})
            for idx, det in enumerate(helmet_data.get('detections', [])):
                self._store_helmet_detection(detection_id, idx + 1, det)
            
            # Store number plate detections
            plate_data = result.get('number_plate_detection', {})
            for plate_det in plate_data.get('plates', []):
                plate_det_id = self._store_plate_detection(
                    detection_id,
                    plate_det
                )
                
                # Store OCR results for this plate
                ocr_data = result.get('ocr_results', {})
                for ocr_result in ocr_data.get('extraction_results', []):
                    if ocr_result.get('plate_id') == plate_det.get('plate_id'):
                        self._store_ocr_result(plate_det_id, ocr_result)
            
            # Store violations if any
            self._store_violations(detection_id, result)
            
            self.conn.commit()
            logger.info(f"Detection results stored with ID: {detection_id}")
            
            return detection_id
        
        except sqlite3.Error as e:
            logger.error(f"Error storing detection: {e}")
            self.conn.rollback()
            return -1
    
    def _store_helmet_detection(
        self,
        detection_id: int,
        person_id: int,
        detection: Dict
    ):
        """Store individual helmet detection"""
        try:
            bbox = detection.get('bbox', (0, 0, 0, 0))
            
            self.cursor.execute('''
                INSERT INTO helmet_detections
                (detection_id, person_id, has_helmet, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_id,
                person_id,
                detection.get('has_helmet'),
                detection.get('confidence'),
                bbox[0], bbox[1], bbox[2], bbox[3],
                detection.get('status')
            ))
        except sqlite3.Error as e:
            logger.error(f"Error storing helmet detection: {e}")
    
    def _store_plate_detection(
        self,
        detection_id: int,
        plate_detection: Dict
    ) -> int:
        """Store number plate detection"""
        try:
            bbox = plate_detection.get('bbox', (0, 0, 0, 0))
            
            self.cursor.execute('''
                INSERT INTO number_plate_detections
                (detection_id, plate_id, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_id,
                plate_detection.get('plate_id'),
                plate_detection.get('confidence'),
                bbox[0], bbox[1], bbox[2], bbox[3],
                plate_detection.get('status')
            ))
            
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error storing plate detection: {e}")
            return -1
    
    def _store_ocr_result(
        self,
        plate_detection_id: int,
        ocr_result: Dict
    ):
        """Store OCR result"""
        try:
            self.cursor.execute('''
                INSERT INTO ocr_results
                (plate_detection_id, extracted_text, cleaned_text,
                 confidence, valid_format)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                plate_detection_id,
                ocr_result.get('extracted_text'),
                ocr_result.get('extracted_text'),  # Cleaned same for now
                ocr_result.get('confidence'),
                ocr_result.get('valid_format', False)
            ))
        except sqlite3.Error as e:
            logger.error(f"Error storing OCR result: {e}")
    
    def _store_violations(self, detection_id: int, result: Dict):
        """Store violation records"""
        try:
            helmet_data = result.get('helmet_detection', {})
            ocr_data = result.get('ocr_results', {})
            
            # Get number plate text
            plate_text = None
            if ocr_data.get('extraction_results'):
                plate_text = ocr_data['extraction_results'][0].get('extracted_text')
            
            # Check for helmet violations
            for idx, det in enumerate(helmet_data.get('detections', [])):
                if not det.get('has_helmet'):
                    # This is a violation
                    self.cursor.execute('''
                        INSERT INTO violations
                        (detection_id, violation_type, severity,
                         image_path, number_plate_text)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        detection_id,
                        'NO_HELMET',
                        'HIGH',
                        result.get('image_path'),
                        plate_text
                    ))
                    
                    logger.info(f"Violation recorded: NO_HELMET - Plate: {plate_text}")
        
        except sqlite3.Error as e:
            logger.error(f"Error storing violations: {e}")
    
    def get_violations(
        self,
        limit: int = 100,
        days: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve violation records
        
        Args:
            limit: Maximum number of records to retrieve
            days: Only return violations from last N days (None = all)
            
        Returns:
            List of violation records
        """
        try:
            query = 'SELECT * FROM violations'
            params = []
            
            if days:
                query += ' WHERE violation_timestamp >= datetime("now", ?)'
                params.append(f'-{days} days')
            
            query += ' ORDER BY violation_timestamp DESC LIMIT ?'
            params.append(limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            return [dict(row) for row in rows]
        
        except sqlite3.Error as e:
            logger.error(f"Error retrieving violations: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {}
            
            # Total detections
            self.cursor.execute('SELECT COUNT(*) as count FROM detection_records')
            stats['total_detections'] = self.cursor.fetchone()['count']
            
            # Total violations
            self.cursor.execute('SELECT COUNT(*) as count FROM violations')
            stats['total_violations'] = self.cursor.fetchone()['count']
            
            # Helmet violations
            self.cursor.execute(
                'SELECT COUNT(*) as count FROM helmet_detections WHERE has_helmet = 0'
            )
            stats['helmet_violations'] = self.cursor.fetchone()['count']
            
            # OCR success rate
            self.cursor.execute(
                'SELECT COUNT(*) as count FROM ocr_results WHERE valid_format = 1'
            )
            stats['valid_ocr_results'] = self.cursor.fetchone()['count']
            
            # Total OCR results
            self.cursor.execute('SELECT COUNT(*) as count FROM ocr_results')
            stats['total_ocr_results'] = self.cursor.fetchone()['count']
            
            if stats['total_ocr_results'] > 0:
                stats['ocr_success_rate'] = (
                    stats['valid_ocr_results'] / stats['total_ocr_results']
                ) * 100
            
            return stats
        
        except sqlite3.Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {}
    
    def export_violations_csv(self, output_file: str = "violations_report.csv"):
        """
        Export violation records to CSV
        
        Args:
            output_file: Path to save CSV file
        """
        try:
            import csv
            
            violations = self.get_violations(limit=999999)
            
            if not violations:
                logger.warning("No violations to export")
                return
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=violations[0].keys()
                )
                writer.writeheader()
                writer.writerows(violations)
            
            logger.info(f"Violations exported to: {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting violations: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

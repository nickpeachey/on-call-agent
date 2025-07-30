"""
Log Archive Service - Manages processed log entries to prevent duplicate processing
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Set, Dict, Any, Optional
from pathlib import Path
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class LogArchiveService:
    """Service for archiving processed log entries to prevent duplicate processing"""
    
    def __init__(self, archive_dir: str = "data/log_archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for recent processed logs (last 24 hours)
        self.processed_cache: Set[str] = set()
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Archive settings
        self.max_cache_age_hours = 24
        self.archive_file_prefix = "processed_logs"
        self.cleanup_interval_hours = 6
        
        # Load recent processed logs from disk
        self._load_recent_processed_logs()
        
        logger.info("Log archive service initialized", 
                   archive_dir=str(self.archive_dir),
                   cached_entries=len(self.processed_cache))
    
    def _generate_log_hash(self, log_entry: Dict[str, Any]) -> str:
        """Generate a unique hash for a log entry to identify duplicates"""
        # Create a normalized representation of the log entry
        # Include timestamp, message, service, and level for uniqueness
        key_fields = {
            'timestamp': log_entry.get('timestamp'),
            'message': log_entry.get('message', ''),
            'service': log_entry.get('service', ''),
            'level': log_entry.get('level', ''),
            'source': log_entry.get('source', '')  # Source file or system
        }
        
        # Create consistent string representation
        normalized = json.dumps(key_fields, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def is_already_processed(self, log_entry: Dict[str, Any]) -> bool:
        """Check if a log entry has already been processed"""
        log_hash = self._generate_log_hash(log_entry)
        
        # Clean up old cache entries first
        self._cleanup_cache()
        
        is_processed = log_hash in self.processed_cache
        
        if is_processed:
            logger.debug("Log entry already processed", 
                        log_hash=log_hash[:16],
                        message_preview=log_entry.get('message', '')[:50])
        
        return is_processed
    
    def mark_as_processed(self, log_entry: Dict[str, Any], incident_created: bool = False):
        """Mark a log entry as processed"""
        log_hash = self._generate_log_hash(log_entry)
        current_time = datetime.utcnow()
        
        # Add to cache
        self.processed_cache.add(log_hash)
        self.cache_timestamps[log_hash] = current_time
        
        # Archive to disk
        self._archive_processed_entry(log_hash, log_entry, incident_created, current_time)
        
        logger.debug("Marked log entry as processed",
                    log_hash=log_hash[:16],
                    incident_created=incident_created,
                    message_preview=log_entry.get('message', '')[:50])
    
    def _archive_processed_entry(self, log_hash: str, log_entry: Dict[str, Any], 
                                incident_created: bool, timestamp: datetime):
        """Archive a processed log entry to disk"""
        try:
            # Create daily archive file
            date_str = timestamp.strftime('%Y-%m-%d')
            archive_file = self.archive_dir / f"{self.archive_file_prefix}_{date_str}.jsonl"
            
            # Archive entry
            archive_entry = {
                'hash': log_hash,
                'timestamp': timestamp.isoformat(),
                'processed_at': timestamp.isoformat(),
                'incident_created': incident_created,
                'log_entry': {
                    'timestamp': log_entry.get('timestamp'),
                    'message': log_entry.get('message', ''),
                    'service': log_entry.get('service', ''),
                    'level': log_entry.get('level', ''),
                    'source': log_entry.get('source', '')
                }
            }
            
            # Append to archive file (JSONL format)
            with open(archive_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(archive_entry) + '\n')
                
        except Exception as e:
            logger.error("Failed to archive processed entry", 
                        log_hash=log_hash[:16], 
                        error=str(e))
    
    def _load_recent_processed_logs(self):
        """Load recently processed logs from disk archive"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.max_cache_age_hours)
            
            # Look for archive files from the last few days
            for days_back in range(3):  # Check last 3 days
                date_str = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                archive_file = self.archive_dir / f"{self.archive_file_prefix}_{date_str}.jsonl"
                
                if archive_file.exists():
                    self._load_archive_file(archive_file, cutoff_time)
            
            logger.info("Loaded recent processed logs from archive",
                       entries_loaded=len(self.processed_cache))
                       
        except Exception as e:
            logger.error("Failed to load recent processed logs", error=str(e))
    
    def _load_archive_file(self, archive_file: Path, cutoff_time: datetime):
        """Load entries from a specific archive file"""
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        processed_at = datetime.fromisoformat(entry['processed_at'])
                        
                        # Only load recent entries
                        if processed_at > cutoff_time:
                            log_hash = entry['hash']
                            self.processed_cache.add(log_hash)
                            self.cache_timestamps[log_hash] = processed_at
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning("Invalid archive entry", 
                                     file=str(archive_file), 
                                     error=str(e))
                        
        except Exception as e:
            logger.error("Failed to load archive file", 
                        file=str(archive_file), 
                        error=str(e))
    
    def _cleanup_cache(self):
        """Remove old entries from the in-memory cache"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_cache_age_hours)
        
        # Find expired entries
        expired_hashes = [
            log_hash for log_hash, timestamp in self.cache_timestamps.items()
            if timestamp < cutoff_time
        ]
        
        # Remove expired entries
        for log_hash in expired_hashes:
            self.processed_cache.discard(log_hash)
            self.cache_timestamps.pop(log_hash, None)
        
        if expired_hashes:
            logger.debug("Cleaned up expired cache entries", 
                        expired_count=len(expired_hashes),
                        remaining_count=len(self.processed_cache))
    
    async def cleanup_old_archives(self, max_age_days: int = 30):
        """Clean up old archive files"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Find old archive files
            old_files = []
            for archive_file in self.archive_dir.glob(f"{self.archive_file_prefix}_*.jsonl"):
                try:
                    # Extract date from filename
                    date_part = archive_file.stem.split('_')[-1]  # Get YYYY-MM-DD part
                    file_date = datetime.strptime(date_part, '%Y-%m-%d')
                    
                    if file_date < cutoff_date:
                        old_files.append(archive_file)
                        
                except ValueError:
                    logger.warning("Invalid archive filename format", 
                                 filename=archive_file.name)
            
            # Remove old files
            for old_file in old_files:
                try:
                    old_file.unlink()
                    logger.info("Removed old archive file", filename=old_file.name)
                except Exception as e:
                    logger.error("Failed to remove old archive file", 
                               filename=old_file.name, 
                               error=str(e))
            
            if old_files:
                logger.info("Cleaned up old archive files", 
                           files_removed=len(old_files),
                           max_age_days=max_age_days)
                           
        except Exception as e:
            logger.error("Failed to cleanup old archives", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the archive service"""
        return {
            'cached_entries': len(self.processed_cache),
            'cache_age_hours': self.max_cache_age_hours,
            'archive_dir': str(self.archive_dir),
            'archive_files': len(list(self.archive_dir.glob(f"{self.archive_file_prefix}_*.jsonl"))),
            'oldest_cached_entry': min(self.cache_timestamps.values()).isoformat() if self.cache_timestamps else None,
            'newest_cached_entry': max(self.cache_timestamps.values()).isoformat() if self.cache_timestamps else None
        }
    
    def force_refresh_cache(self):
        """Force refresh the cache from disk archives"""
        logger.info("Force refreshing archive cache")
        
        # Clear current cache
        self.processed_cache.clear()
        self.cache_timestamps.clear()
        
        # Reload from disk
        self._load_recent_processed_logs()
        
        logger.info("Archive cache refreshed", 
                   entries_loaded=len(self.processed_cache))

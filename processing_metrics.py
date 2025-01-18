import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ProcessingMetrics:
    """Class to track processing metrics with accurate time estimation"""
    start_time: float = field(default_factory=time.time)
    processing_times: List[float] = field(default_factory=list)
    batch_times: List[float] = field(default_factory=list)
    total_documents: int = 0
    processed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    current_batch: int = 0
    pause_start_time: Optional[float] = None
    total_pause_time: float = 0.0
    batch_size: int = 10
    
    def add_processing_time(self, duration: float) -> None:
        """Add a new processing time and maintain rolling window"""
        self.processing_times.append(duration)
        # Keep only the last 50 times for more accurate rolling average
        if len(self.processing_times) > 50:
            self.processing_times.pop(0)
    
    def add_batch_time(self, duration: float) -> None:
        """Add a new batch processing time"""
        self.batch_times.append(duration)
        if len(self.batch_times) > 10:  # Keep last 10 batch times
            self.batch_times.pop(0)
    
    def pause(self) -> None:
        """Record start of a pause"""
        if not self.pause_start_time:
            self.pause_start_time = time.time()
    
    def resume(self) -> None:
        """Record end of a pause and update total pause time"""
        if self.pause_start_time:
            self.total_pause_time += time.time() - self.pause_start_time
            self.pause_start_time = None
    
    def get_elapsed_time(self) -> float:
        """Get actual processing time excluding pauses"""
        current_pause = time.time() - self.pause_start_time if self.pause_start_time else 0
        return time.time() - self.start_time - self.total_pause_time - current_pause
    
    def get_processing_rate(self) -> Dict[str, float]:
        """Calculate current processing rates"""
        if not self.processing_times:
            return {"docs_per_second": 0, "docs_per_minute": 0, "seconds_per_doc": 0}
        
        # Use different averaging methods for more accuracy
        recent_times = self.processing_times[-10:] if len(self.processing_times) > 10 else self.processing_times
        
        # Calculate rates using different methods
        mean_time = np.mean(recent_times)
        median_time = np.median(recent_times)
        
        # Use weighted average favoring recent times
        weights = np.linspace(0.5, 1.0, len(recent_times))
        weighted_time = np.average(recent_times, weights=weights)
        
        # Combine methods for more stable estimate
        avg_time = (mean_time + median_time + weighted_time) / 3
        
        return {
            "docs_per_second": 1 / avg_time,
            "docs_per_minute": 60 / avg_time,
            "seconds_per_doc": avg_time
        }
    
    def get_estimated_completion(self) -> Dict[str, any]:
        """Calculate estimated completion time and related metrics"""
        if not self.processing_times or self.processed_count == 0:
            return {
                "time_remaining": "Calculating...",
                "estimated_completion": "Calculating...",
                "accuracy": "low"
            }
        
        # Calculate remaining documents
        remaining_docs = self.total_documents - (self.processed_count + self.failed_count + self.skipped_count)
        
        # Get processing rates
        rates = self.get_processing_rate()
        
        # Calculate time remaining using multiple methods
        if rates["docs_per_second"] > 0:
            # Method 1: Based on recent individual document times
            time_remaining_1 = remaining_docs * rates["seconds_per_doc"]
            
            # Method 2: Based on overall progress rate
            elapsed_time = self.get_elapsed_time()
            if elapsed_time > 0:
                overall_rate = self.processed_count / elapsed_time
                time_remaining_2 = remaining_docs / overall_rate
            else:
                time_remaining_2 = time_remaining_1
            
            # Method 3: Based on batch times if available
            if self.batch_times:
                avg_batch_time = np.mean(self.batch_times)
                remaining_batches = (remaining_docs + self.batch_size - 1) // self.batch_size
                time_remaining_3 = remaining_batches * avg_batch_time
            else:
                time_remaining_3 = time_remaining_1
            
            # Weighted average of all methods
            time_remaining = (time_remaining_1 * 0.4 + 
                            time_remaining_2 * 0.4 + 
                            time_remaining_3 * 0.2)
            
            # Calculate accuracy confidence
            variation_coefficient = np.std(self.processing_times) / np.mean(self.processing_times)
            accuracy = "high" if variation_coefficient < 0.2 else "medium" if variation_coefficient < 0.5 else "low"
            
            # Calculate estimated completion time
            completion_time = datetime.now() + timedelta(seconds=int(time_remaining))
            
            # Format time remaining
            if time_remaining > 3600:
                time_str = f"{time_remaining/3600:.1f} hours"
            elif time_remaining > 60:
                time_str = f"{time_remaining/60:.1f} minutes"
            else:
                time_str = f"{time_remaining:.0f} seconds"
            
            return {
                "time_remaining": time_str,
                "estimated_completion": completion_time.strftime("%I:%M %p"),
                "accuracy": accuracy
            }
        else:
            return {
                "time_remaining": "Calculating...",
                "estimated_completion": "Calculating...",
                "accuracy": "low"
            }
    
    def get_progress_stats(self) -> Dict[str, any]:
        """Get comprehensive progress statistics"""
        total_processed = self.processed_count + self.failed_count + self.skipped_count
        progress = (total_processed / self.total_documents) if self.total_documents > 0 else 0
        
        rates = self.get_processing_rate()
        completion_estimate = self.get_estimated_completion()
        
        return {
            "progress_percentage": progress * 100,
            "total_documents": self.total_documents,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "remaining": self.total_documents - total_processed,
            "processing_rate": rates,
            "estimated_completion": completion_estimate,
            "elapsed_time": self.get_elapsed_time()
        } 
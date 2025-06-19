# robomerge/robomerge/ops_dashboard_backend.py

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue

@dataclass
class OperatorMetrics:
    """Metrics for individual operator performance."""
    operator_id: str
    name: str
    demonstrations_completed: int
    demonstrations_failed: int
    avg_data_quality: float
    avg_completion_time: float
    current_task: Optional[str]
    last_active: datetime
    quality_trend: List[float]  # Last 10 quality scores
    
    @property
    def success_rate(self) -> float:
        total = self.demonstrations_completed + self.demonstrations_failed
        return self.demonstrations_completed / total if total > 0 else 0.0
    
    @property
    def performance_score(self) -> float:
        """Combined performance score (0-100)."""
        return (self.success_rate * 0.4 + 
                self.avg_data_quality * 0.4 + 
                (1.0 / max(self.avg_completion_time, 0.1)) * 0.2) * 100

@dataclass
class ProcessingQueueItem:
    """Item in the processing queue."""
    task_id: str
    operator_id: str
    task_type: str
    priority: int
    submitted_at: datetime
    estimated_duration: float
    status: str  # 'queued', 'processing', 'completed', 'failed'
    
@dataclass
class QualityAlert:
    """Quality alert for the monitoring system."""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    operator_id: Optional[str]
    task_id: Optional[str]
    timestamp: datetime
    resolved: bool = False

class OperationsDashboard:
    """Operations dashboard for monitoring robot data collection and processing."""
    
    def __init__(self, max_history_hours: int = 24):
        self.max_history_hours = max_history_hours
        
        # Core data stores
        self.operators: Dict[str, OperatorMetrics] = {}
        self.processing_queue: List[ProcessingQueueItem] = []
        self.quality_alerts: List[QualityAlert] = []
        
        # Historical data (time-series)
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.throughput_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=1000)
        
        # Real-time monitoring
        self.real_time_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Thresholds for alerts
        self.quality_thresholds = {
            'critical': 0.3,
            'high': 0.5,
            'medium': 0.7,
            'low': 0.8
        }
        
        self.performance_thresholds = {
            'critical': 20.0,
            'high': 40.0, 
            'medium': 60.0,
            'low': 80.0
        }
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def add_operator(self, operator_id: str, name: str) -> None:
        """Add a new operator to the system."""
        self.operators[operator_id] = OperatorMetrics(
            operator_id=operator_id,
            name=name,
            demonstrations_completed=0,
            demonstrations_failed=0,
            avg_data_quality=0.0,
            avg_completion_time=0.0,
            current_task=None,
            last_active=datetime.now(),
            quality_trend=[]
        )
    
    def update_operator_metrics(self, 
                               operator_id: str,
                               demonstration_result: Dict[str, Any]) -> None:
        """Update operator metrics with new demonstration result."""
        if operator_id not in self.operators:
            self.add_operator(operator_id, f"Operator_{operator_id}")
        
        operator = self.operators[operator_id]
        
        # Update basic counters
        if demonstration_result.get('success', False):
            operator.demonstrations_completed += 1
        else:
            operator.demonstrations_failed += 1
        
        # Update quality metrics
        quality_score = demonstration_result.get('quality_score', 0.0)
        operator.quality_trend.append(quality_score)
        if len(operator.quality_trend) > 10:
            operator.quality_trend.pop(0)
        
        # Recalculate averages
        operator.avg_data_quality = np.mean(operator.quality_trend)
        
        # Update completion time
        completion_time = demonstration_result.get('completion_time', 0.0)
        if completion_time > 0:
            # Exponential moving average
            alpha = 0.1
            if operator.avg_completion_time == 0:
                operator.avg_completion_time = completion_time
            else:
                operator.avg_completion_time = (alpha * completion_time + 
                                              (1 - alpha) * operator.avg_completion_time)
        
        operator.last_active = datetime.now()
        
        # Check for quality alerts
        self._check_quality_alerts(operator_id, quality_score)
       
       # Add to historical data
        self._add_to_history(operator_id, demonstration_result)
   
    def add_to_queue(self, 
                    task_id: str,
                    operator_id: str, 
                    task_type: str,
                    priority: int = 1,
                    estimated_duration: float = 300.0) -> None:
        """Add item to processing queue."""
        queue_item = ProcessingQueueItem(
            task_id=task_id,
            operator_id=operator_id,
            task_type=task_type,
            priority=priority,
            submitted_at=datetime.now(),
            estimated_duration=estimated_duration,
            status='queued'
        )
        
        # Insert based on priority
        inserted = False
        for i, item in enumerate(self.processing_queue):
            if priority > item.priority:
                self.processing_queue.insert(i, queue_item)
                inserted = True
                break
        
        if not inserted:
            self.processing_queue.append(queue_item)
    
    def update_queue_status(self, task_id: str, status: str) -> None:
        """Update status of item in processing queue."""
        for item in self.processing_queue:
            if item.task_id == task_id:
                item.status = status
                break
        
        # Remove completed/failed items after some time
        if status in ['completed', 'failed']:
            # Keep for historical analysis, but could implement cleanup
            pass
    
    def get_operator_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get operator leaderboard sorted by performance score."""
        operators = list(self.operators.values())
        operators.sort(key=lambda x: x.performance_score, reverse=True)
        
        leaderboard = []
        for i, op in enumerate(operators[:limit]):
            leaderboard.append({
                'rank': i + 1,
                'operator_id': op.operator_id,
                'name': op.name,
                'performance_score': round(op.performance_score, 1),
                'success_rate': round(op.success_rate * 100, 1),
                'avg_quality': round(op.avg_data_quality, 2),
                'completed': op.demonstrations_completed,
                'failed': op.demonstrations_failed,
                'current_task': op.current_task,
                'last_active': op.last_active.isoformat()
            })
        
        return leaderboard
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current processing queue status."""
        status_counts = defaultdict(int)
        total_wait_time = 0.0
        high_priority_count = 0
        
        now = datetime.now()
        
        for item in self.processing_queue:
            status_counts[item.status] += 1
            
            if item.status == 'queued':
                wait_time = (now - item.submitted_at).total_seconds()
                total_wait_time += wait_time
                
                if item.priority > 3:
                    high_priority_count += 1
        
        avg_wait_time = (total_wait_time / max(status_counts['queued'], 1)) / 60  # minutes
        
        return {
            'total_items': len(self.processing_queue),
            'queued': status_counts['queued'],
            'processing': status_counts['processing'],
            'completed_today': self._count_completed_today(),
            'failed_today': self._count_failed_today(),
            'avg_wait_time_minutes': round(avg_wait_time, 1),
            'high_priority_waiting': high_priority_count,
            'estimated_completion_time': self._estimate_queue_completion()
        }
    
    def get_quality_trends(self, hours: int = 24) -> Dict[str, List]:
        """Get quality trends over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        trends = {
            'timestamps': [],
            'overall_quality': [],
            'throughput': [],
            'error_rate': []
        }
        
        # Process historical data
        for timestamp, data in self.quality_history.items():
            if datetime.fromisoformat(timestamp) >= cutoff_time:
                trends['timestamps'].append(timestamp)
                trends['overall_quality'].append(data.get('quality', 0.0))
                trends['throughput'].append(data.get('throughput', 0))
                trends['error_rate'].append(data.get('error_rate', 0.0))
        
        return trends
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active (unresolved) alerts."""
        active_alerts = [alert for alert in self.quality_alerts if not alert.resolved]
        
        # Sort by severity and timestamp
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        active_alerts.sort(
            key=lambda x: (severity_order.get(x.severity, 0), x.timestamp),
            reverse=True
        )
        
        return [asdict(alert) for alert in active_alerts]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.quality_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        return {
            'overview': {
                'total_operators': len(self.operators),
                'active_operators': self._count_active_operators(),
                'total_demonstrations_today': self._count_demonstrations_today(),
                'avg_quality_today': self._get_avg_quality_today(),
                'success_rate_today': self._get_success_rate_today()
            },
            'queue': self.get_queue_status(),
            'alerts': {
                'critical': len([a for a in self.quality_alerts if a.severity == 'critical' and not a.resolved]),
                'high': len([a for a in self.quality_alerts if a.severity == 'high' and not a.resolved]),
                'total_active': len([a for a in self.quality_alerts if not a.resolved])
            },
            'top_performers': self.get_operator_leaderboard(5),
            'last_updated': datetime.now().isoformat()
        }
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Process any queued updates
                while not self.real_time_queue.empty():
                    update = self.real_time_queue.get_nowait()
                    self._process_real_time_update(update)
                
                # Perform periodic checks
                self._check_system_health()
                self._cleanup_old_data()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _check_quality_alerts(self, operator_id: str, quality_score: float) -> None:
        """Check if quality score triggers any alerts."""
        severity = None
        
        if quality_score <= self.quality_thresholds['critical']:
            severity = 'critical'
        elif quality_score <= self.quality_thresholds['high']:
            severity = 'high'
        elif quality_score <= self.quality_thresholds['medium']:
            severity = 'medium'
        elif quality_score <= self.quality_thresholds['low']:
            severity = 'low'
        
        if severity:
            alert = QualityAlert(
                alert_id=f"quality_{operator_id}_{int(time.time())}",
                severity=severity,
                message=f"Low data quality ({quality_score:.2f}) from operator {operator_id}",
                operator_id=operator_id,
                task_id=None,
                timestamp=datetime.now()
            )
            self.quality_alerts.append(alert)
    
    def _add_to_history(self, operator_id: str, result: Dict[str, Any]) -> None:
        """Add result to historical tracking."""
        timestamp = datetime.now().isoformat()
        
        self.quality_history[timestamp] = {
            'operator_id': operator_id,
            'quality': result.get('quality_score', 0.0),
            'success': result.get('success', False),
            'completion_time': result.get('completion_time', 0.0)
        }
    
    def _count_active_operators(self) -> int:
        """Count operators active in last hour."""
        cutoff = datetime.now() - timedelta(hours=1)
        return len([op for op in self.operators.values() if op.last_active >= cutoff])
    
    def _count_demonstrations_today(self) -> int:
        """Count total demonstrations completed today."""
        return sum(op.demonstrations_completed + op.demonstrations_failed 
                    for op in self.operators.values())
    
    def _get_avg_quality_today(self) -> float:
        """Get average quality score for today."""
        today = datetime.now().date()
        today_qualities = []
        
        for timestamp, data in self.quality_history.items():
            if datetime.fromisoformat(timestamp).date() == today:
                today_qualities.append(data.get('quality', 0.0))
        
        return np.mean(today_qualities) if today_qualities else 0.0
    
    def _get_success_rate_today(self) -> float:
        """Get success rate for today."""
        today = datetime.now().date()
        successes = 0
        total = 0
        
        for timestamp, data in self.quality_history.items():
            if datetime.fromisoformat(timestamp).date() == today:
                total += 1
                if data.get('success', False):
                    successes += 1
        
        return (successes / total) if total > 0 else 0.0
    
    def _count_completed_today(self) -> int:
        """Count completed tasks today."""
        today = datetime.now().date()
        return len([item for item in self.processing_queue 
                    if item.status == 'completed' and item.submitted_at.date() == today])
    
    def _count_failed_today(self) -> int:
        """Count failed tasks today."""
        today = datetime.now().date()
        return len([item for item in self.processing_queue 
                    if item.status == 'failed' and item.submitted_at.date() == today])
    
    def _estimate_queue_completion(self) -> str:
        """Estimate when current queue will be completed."""
        queued_items = [item for item in self.processing_queue if item.status == 'queued']
        total_time = sum(item.estimated_duration for item in queued_items)
        
        if total_time == 0:
            return "Queue empty"
        
        completion_time = datetime.now() + timedelta(seconds=total_time)
        return completion_time.strftime("%H:%M")
    
    def _process_real_time_update(self, update: Dict[str, Any]) -> None:
        """Process real-time update from queue."""
        update_type = update.get('type')
        
        if update_type == 'operator_result':
            self.update_operator_metrics(
                update['operator_id'], 
                update['result']
            )
        elif update_type == 'queue_update':
            self.update_queue_status(
                update['task_id'], 
                update['status']
            )
    
    def _check_system_health(self) -> None:
        """Perform system health checks."""
        # Check for operators who haven't been active
        inactive_threshold = datetime.now() - timedelta(hours=2)
        
        for operator in self.operators.values():
            if operator.last_active < inactive_threshold and operator.current_task:
                alert = QualityAlert(
                    alert_id=f"inactive_{operator.operator_id}_{int(time.time())}",
                    severity='medium',
                    message=f"Operator {operator.name} inactive for 2+ hours with active task",
                    operator_id=operator.operator_id,
                    task_id=operator.current_task,
                    timestamp=datetime.now()
                )
                self.quality_alerts.append(alert)
        
        # Check for queue bottlenecks
        long_waiting = [item for item in self.processing_queue 
                        if item.status == 'queued' and 
                        (datetime.now() - item.submitted_at).total_seconds() > 1800]  # 30 min
        
        if len(long_waiting) > 5:
            alert = QualityAlert(
                alert_id=f"queue_bottleneck_{int(time.time())}",
                severity='high',
                message=f"Queue bottleneck: {len(long_waiting)} items waiting >30min",
                operator_id=None,
                task_id=None,
                timestamp=datetime.now()
            )
            self.quality_alerts.append(alert)
    
    def _cleanup_old_data(self) -> None:
        """Clean up old historical data."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
        
        # Clean up quality history
        old_timestamps = [ts for ts in self.quality_history.keys() 
                            if datetime.fromisoformat(ts) < cutoff_time]
        
        for ts in old_timestamps:
            del self.quality_history[ts]
        
        # Clean up old alerts (resolved ones older than 24 hours)
        alert_cutoff = datetime.now() - timedelta(hours=24)
        self.quality_alerts = [alert for alert in self.quality_alerts 
                                if not (alert.resolved and alert.timestamp < alert_cutoff)]
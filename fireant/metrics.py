"""
FireAnt performance metrics collection.
Provides comprehensive metrics collection and analysis for agents and flows.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Represents a single metric value."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class HistogramBucket:
    """Represents a histogram bucket."""
    upper_bound: float
    count: int


@dataclass
class HistogramMetric:
    """Represents a histogram metric with buckets."""
    name: str
    buckets: List[HistogramBucket]
    count: int
    sum: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """Registry for collecting and managing metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(lambda: deque(maxlen=max_history))
        self._histograms: Dict[str, List[HistogramMetric]] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
    
    def counter(self, name: str, value: int = 1, labels: Dict[str, str] = None, timestamp: float = None):
        """Record a counter metric."""
        if labels is None:
            labels = {}
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Update counter total
            key = self._make_key(name, labels)
            self._counters[key] += value
            
            # Record metric value
            metric = MetricValue(
                name=name,
                value=self._counters[key],
                metric_type=MetricType.COUNTER,
                timestamp=timestamp,
                labels=labels
            )
            self._metrics[name].append(metric)
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None, timestamp: float = None):
        """Record a gauge metric."""
        if labels is None:
            labels = {}
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Update gauge value
            key = self._make_key(name, labels)
            self._gauges[key] = value
            
            # Record metric value
            metric = MetricValue(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                labels=labels
            )
            self._metrics[name].append(metric)
    
    def histogram(self, name: str, value: float, buckets: List[float] = None, 
                  labels: Dict[str, str] = None, timestamp: float = None):
        """Record a histogram metric."""
        if buckets is None:
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        if labels is None:
            labels = {}
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            key = self._make_key(name, labels)
            
            # Initialize histogram if needed
            if key not in self._histograms or not self._histograms[key]:
                # Create initial histogram
                hist_buckets = [HistogramBucket(upper_bound=b, count=0) for b in buckets] + \
                              [HistogramBucket(upper_bound=float('inf'), count=0)]
                
                histogram = HistogramMetric(
                    name=name,
                    buckets=hist_buckets,
                    count=0,
                    sum=0.0,
                    timestamp=timestamp,
                    labels=labels
                )
                self._histograms[key] = histogram
            else:
                histogram = self._histograms[key][-1]
            
            # Update histogram
            histogram.count += 1
            histogram.sum += value
            
            # Find appropriate bucket
            for bucket in histogram.buckets:
                if value <= bucket.upper_bound:
                    bucket.count += 1
                    break
            
            self._histograms[key].append(histogram)
    
    def timer(self, name: str, duration: float, labels: Dict[str, str] = None, timestamp: float = None):
        """Record a timer metric."""
        if labels is None:
            labels = {}
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Store timer value
            key = self._make_key(name, labels)
            self._timers[key].append(duration)
            
            # Record as histogram
            self.histogram(f"{name}_seconds", duration, labels=labels, timestamp=timestamp)
            
            # Record as gauge for current value
            self.gauge(f"{name}_last", duration, labels=labels, timestamp=timestamp)
    
    def get_metrics(self, name: str = None) -> List[MetricValue]:
        """Get metrics by name."""
        with self._lock:
            if name:
                return list(self._metrics.get(name, []))
            else:
                # Return all metrics
                all_metrics = []
                for metrics in self._metrics.values():
                    all_metrics.extend(metrics)
                return all_metrics
    
    def get_histograms(self, name: str = None) -> List[HistogramMetric]:
        """Get histogram metrics by name."""
        with self._lock:
            if name:
                return list(self._histograms.get(name, []))
            else:
                # Return all histograms
                all_histograms = []
                for histograms in self._histograms.values():
                    all_histograms.extend(histograms)
                return all_histograms
    
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics."""
        if labels is None:
            labels = {}
        
        key = self._make_key(name, labels)
        
        with self._lock:
            if key not in self._timers or not self._timers[key]:
                return {}
            
            durations = list(self._timers[key])
            
            if not durations:
                return {}
            
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "mean": sum(durations) / len(durations),
                "sum": sum(durations)
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._histograms.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a key from name and labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        with self._lock:
            lines = []
            
            # Export counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                label_str = self._format_labels(labels)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{label_str} {value}")
            
            # Export gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                label_str = self._format_labels(labels)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{label_str} {value}")
            
            # Export histograms
            for histogram in self.get_histograms():
                label_str = self._format_labels(histogram.labels)
                lines.append(f"# TYPE {histogram.name} histogram")
                
                # Add bucket counts
                cumulative = 0
                for bucket in histogram.buckets:
                    cumulative += bucket.count
                    bucket_labels = {**histogram.labels, "le": str(bucket.upper_bound)}
                    bucket_label_str = self._format_labels(bucket_labels)
                    lines.append(f"{histogram.name}_bucket{bucket_label_str} {cumulative}")
                
                # Add count and sum
                lines.append(f"{histogram.name}_count{label_str} {histogram.count}")
                lines.append(f"{histogram.name}_sum{label_str} {histogram.sum}")
            
            return "\n".join(lines)
    
    def _parse_key(self, key: str) -> tuple:
        """Parse a key into name and labels."""
        if "{" in key and key.endswith("}"):
            name = key[:key.index("{")]
            label_str = key[key.index("{")+1:-1]
            labels = {}
            
            for pair in label_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip()
            
            return name, labels
        else:
            return key, {}
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus export."""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"


class PerformanceMetrics:
    """High-level performance metrics for FireAnt."""
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()
        self._start_time = time.time()
    
    def record_agent_execution(self, agent_name: str, execution_time: float, 
                              success: bool, retry_count: int = 0):
        """Record agent execution metrics."""
        labels = {"agent": agent_name, "success": str(success)}
        
        # Record execution time
        self.registry.timer("agent_execution_time_seconds", execution_time, labels)
        
        # Record success/failure
        self.registry.counter("agent_executions_total", 1, labels)
        
        # Record retry count
        if retry_count > 0:
            retry_labels = {"agent": agent_name}
            self.registry.counter("agent_retries_total", retry_count, retry_labels)
    
    def record_flow_execution(self, flow_id: str, execution_time: float, 
                            success: bool, agent_count: int):
        """Record flow execution metrics."""
        labels = {"flow": flow_id, "success": str(success)}
        
        # Record execution time
        self.registry.timer("flow_execution_time_seconds", execution_time, labels)
        
        # Record success/failure
        self.registry.counter("flow_executions_total", 1, labels)
        
        # Record agent count
        self.registry.gauge("flow_agent_count", agent_count, {"flow": flow_id})
    
    def record_memory_usage(self, component: str, memory_bytes: int):
        """Record memory usage metrics."""
        self.registry.gauge("memory_usage_bytes", memory_bytes, {"component": component})
    
    def record_custom_metric(self, name: str, value: Union[int, float], 
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Dict[str, str] = None):
        """Record a custom metric."""
        if metric_type == MetricType.COUNTER:
            self.registry.counter(name, int(value), labels)
        elif metric_type == MetricType.GAUGE:
            self.registry.gauge(name, float(value), labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.registry.histogram(name, float(value), labels=labels)
        elif metric_type == MetricType.TIMER:
            self.registry.timer(name, float(value), labels=labels)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_metrics": len(self.registry.get_metrics()),
            "total_histograms": len(self.registry.get_histograms()),
            "counters": dict(self.registry._counters),
            "gauges": dict(self.registry._gauges)
        }
    
    def export_json(self) -> str:
        """Export metrics as JSON."""
        summary = self.get_summary()
        
        # Add detailed metrics
        summary["metrics"] = [
            {
                "name": m.name,
                "value": m.value,
                "type": m.metric_type.value,
                "timestamp": m.timestamp,
                "labels": m.labels,
                "unit": m.unit
            }
            for m in self.registry.get_metrics()
        ]
        
        summary["histograms"] = [
            {
                "name": h.name,
                "count": h.count,
                "sum": h.sum,
                "timestamp": h.timestamp,
                "labels": h.labels,
                "buckets": [
                    {"upper_bound": b.upper_bound, "count": b.count}
                    for b in h.buckets
                ]
            }
            for h in self.registry.get_histograms()
        ]
        
        return json.dumps(summary, indent=2, default=str)


# Context manager for timing operations
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: PerformanceMetrics, name: str, 
                 labels: Dict[str, str] = None, record_on_success: bool = True):
        self.metrics = metrics
        self.name = name
        self.labels = labels or {}
        self.record_on_success = record_on_success
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            
            # Only record if successful (or always if configured)
            if self.record_on_success or self.success:
                final_labels = self.labels.copy()
                final_labels["success"] = str(self.success)
                self.metrics.registry.timer(self.name, duration, final_labels)
    
    def set_failure(self):
        """Mark the timed operation as failed."""
        self.success = False


# Global metrics instance
_default_metrics: Optional[PerformanceMetrics] = None


def get_default_metrics() -> PerformanceMetrics:
    """Get the default metrics instance."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = PerformanceMetrics()
    return _default_metrics


def set_default_metrics(metrics: PerformanceMetrics):
    """Set the default metrics instance."""
    global _default_metrics
    _default_metrics = metrics


def timer(name: str, labels: Dict[str, str] = None, record_on_success: bool = True):
    """Decorator for timing functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_default_metrics()
            with Timer(metrics, name, labels, record_on_success):
                return func(*args, **kwargs)
        return wrapper
    
    return decorator


def async_timer(name: str, labels: Dict[str, str] = None, record_on_success: bool = True):
    """Decorator for timing async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics = get_default_metrics()
            with Timer(metrics, name, labels, record_on_success):
                return await func(*args, **kwargs)
        return wrapper
    
    return decorator
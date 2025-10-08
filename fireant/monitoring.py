"""
FireAnt monitoring and logging capabilities.
Provides comprehensive logging, metrics collection, and performance monitoring for agents.
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import threading


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AgentMetric:
    """Metric data for agent execution."""
    agent_name: str
    execution_id: str
    status: str
    start_time: float
    end_time: float
    execution_time: float
    input_size: int
    output_size: int
    error_message: Optional[str] = None
    retry_count: int = 0
    memory_usage: Optional[float] = None


@dataclass
class FlowMetric:
    """Metric data for flow execution."""
    flow_id: str
    execution_id: str
    status: str
    start_time: float
    end_time: float
    execution_time: float
    agent_count: int
    successful_agents: int
    failed_agents: int
    total_retries: int


class MetricsCollector:
    """Collects and stores metrics for agents and flows."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.agent_metrics: deque = deque(maxlen=max_history)
        self.flow_metrics: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
    
    def record_agent_metric(self, metric: AgentMetric):
        """Record an agent execution metric."""
        with self._lock:
            self.agent_metrics.append(metric)
    
    def record_flow_metric(self, metric: FlowMetric):
        """Record a flow execution metric."""
        with self._lock:
            self.flow_metrics.append(metric)
    
    def get_agent_metrics(self, agent_name: Optional[str] = None, 
                         limit: int = 100) -> List[AgentMetric]:
        """Get metrics for agents, optionally filtered by name."""
        with self._lock:
            metrics = list(self.agent_metrics)
            if agent_name:
                metrics = [m for m in metrics if m.agent_name == agent_name]
            return metrics[-limit:]
    
    def get_flow_metrics(self, limit: int = 100) -> List[FlowMetric]:
        """Get metrics for flows."""
        with self._lock:
            return list(self.flow_metrics)[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self._lock:
            if not self.agent_metrics:
                return {"message": "No metrics available"}
            
            agent_metrics = list(self.agent_metrics)
            flow_metrics = list(self.flow_metrics)
            
            # Agent statistics
            total_agents = len(agent_metrics)
            successful_agents = sum(1 for m in agent_metrics if m.status == "success")
            failed_agents = total_agents - successful_agents
            avg_execution_time = sum(m.execution_time for m in agent_metrics) / total_agents
            total_retries = sum(m.retry_count for m in agent_metrics)
            
            # Flow statistics
            total_flows = len(flow_metrics)
            successful_flows = sum(1 for m in flow_metrics if m.status == "success")
            avg_flow_time = sum(m.execution_time for m in flow_metrics) / total_flows if total_flows > 0 else 0
            
            return {
                "agents": {
                    "total": total_agents,
                    "successful": successful_agents,
                    "failed": failed_agents,
                    "success_rate": successful_agents / total_agents if total_agents > 0 else 0,
                    "avg_execution_time": avg_execution_time,
                    "total_retries": total_retries
                },
                "flows": {
                    "total": total_flows,
                    "successful": successful_flows,
                    "success_rate": successful_flows / total_flows if total_flows > 0 else 0,
                    "avg_execution_time": avg_flow_time
                }
            }


class FireAntLogger:
    """Enhanced logger for FireAnt agents and flows."""
    
    def __init__(self, name: str = "fireant", level: LogLevel = LogLevel.INFO, 
                 log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method with context support."""
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {context_str}"
        
        getattr(self.logger, level.value.lower())(message)
    
    def log_agent_start(self, agent_name: str, execution_id: str, inputs: Dict[str, Any]):
        """Log agent execution start."""
        self.info(f"Agent started", 
                 agent=agent_name, 
                 execution_id=execution_id,
                 input_keys=list(inputs.keys()),
                 input_size=len(str(inputs)))
    
    def log_agent_success(self, agent_name: str, execution_id: str, 
                         execution_time: float, outputs: Dict[str, Any]):
        """Log agent execution success."""
        self.info(f"Agent completed successfully",
                 agent=agent_name,
                 execution_id=execution_id,
                 execution_time=f"{execution_time:.3f}s",
                 output_keys=list(outputs.keys()),
                 output_size=len(str(outputs)))
    
    def log_agent_failure(self, agent_name: str, execution_id: str, 
                         execution_time: float, error: Exception):
        """Log agent execution failure."""
        self.error(f"Agent execution failed",
                  agent=agent_name,
                  execution_id=execution_id,
                  execution_time=f"{execution_time:.3f}s",
                  error_type=type(error).__name__,
                  error_message=str(error))
    
    def log_agent_retry(self, agent_name: str, execution_id: str, 
                       attempt: int, error: Exception):
        """Log agent retry attempt."""
        self.warning(f"Agent retry attempt",
                    agent=agent_name,
                    execution_id=execution_id,
                    attempt=attempt,
                    error_type=type(error).__name__,
                    error_message=str(error))
    
    def log_flow_start(self, flow_id: str, execution_id: str):
        """Log flow execution start."""
        self.info(f"Flow started", flow_id=flow_id, execution_id=execution_id)
    
    def log_flow_success(self, flow_id: str, execution_id: str, execution_time: float):
        """Log flow execution success."""
        self.info(f"Flow completed successfully",
                 flow_id=flow_id,
                 execution_id=execution_id,
                 execution_time=f"{execution_time:.3f}s")
    
    def log_flow_failure(self, flow_id: str, execution_id: str, execution_time: float, error: Exception):
        """Log flow execution failure."""
        self.error(f"Flow execution failed",
                  flow_id=flow_id,
                  execution_id=execution_id,
                  execution_time=f"{execution_time:.3f}s",
                  error_type=type(error).__name__,
                  error_message=str(error))


class MonitoringMixin:
    """Mixin class to add monitoring capabilities to agents and flows."""
    
    # Global metrics collector and logger
    _metrics_collector = MetricsCollector()
    _logger = FireAntLogger()
    
    @classmethod
    def set_logger(cls, logger: FireAntLogger):
        """Set a custom logger for all instances."""
        cls._logger = logger
    
    @classmethod
    def set_metrics_collector(cls, collector: MetricsCollector):
        """Set a custom metrics collector for all instances."""
        cls._metrics_collector = collector
    
    @classmethod
    def get_metrics_collector(cls) -> MetricsCollector:
        """Get the global metrics collector."""
        return cls._metrics_collector
    
    @classmethod
    def get_logger(cls) -> FireAntLogger:
        """Get the global logger."""
        return cls._logger
    
    def _get_execution_id(self) -> str:
        """Generate a unique execution ID."""
        return f"{self.name}_{int(time.time() * 1000000)}"
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate the size of data in characters."""
        return len(str(data)) if data is not None else 0


class PerformanceProfiler:
    """Context manager for profiling agent and flow performance."""
    
    def __init__(self, name: str, execution_id: str, logger: FireAntLogger, 
                 metrics_collector: MetricsCollector):
        self.name = name
        self.execution_id = execution_id
        self.logger = logger
        self.metrics_collector = metrics_collector
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Profiler: {self.name} completed",
                           execution_id=self.execution_id,
                           execution_time=f"{execution_time:.3f}s")
        else:
            self.logger.error(f"Profiler: {self.name} failed",
                            execution_id=self.execution_id,
                            execution_time=f"{execution_time:.3f}s",
                            error_type=exc_type.__name__,
                            error_message=str(exc_val))


# Global logger instance
_default_logger: Optional[FireAntLogger] = None


def get_default_logger() -> FireAntLogger:
    """Get the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = FireAntLogger()
    return _default_logger


def set_default_logger(logger: FireAntLogger):
    """Set the default logger."""
    global _default_logger
    _default_logger = logger
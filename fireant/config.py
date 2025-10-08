"""
FireAnt configuration management.
Provides centralized configuration management for agents and flows.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class RetryConfig:
    """Configuration for retry policies."""
    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    enabled: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_history_size: int = 1000
    enable_performance_profiling: bool = True


@dataclass
class PersistenceConfig:
    """Configuration for persistence."""
    enabled: bool = False
    storage_type: str = "file"  # file, memory, database
    storage_dir: str = "fireant_states"
    serializer: str = "json"  # json, pickle
    auto_save: bool = True
    cleanup_old_states: bool = True
    max_state_age_hours: int = 24


@dataclass
class AsyncConfig:
    """Configuration for async operations."""
    enabled: bool = False
    max_concurrent_agents: int = 10
    timeout: Optional[float] = None
    enable_async_circuit_breaker: bool = True


@dataclass
class FireAntConfig:
    """Main FireAnt configuration."""
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    async_config: AsyncConfig = field(default_factory=AsyncConfig)
    
    # Global settings
    debug: bool = False
    environment: str = "development"
    version: str = "1.0.0"
    
    # Custom configuration
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages FireAnt configuration."""
    
    def __init__(self, config: Optional[FireAntConfig] = None):
        self._config = config or FireAntConfig()
        self._config_sources = []
    
    def load_from_file(self, file_path: Union[str, Path], format: Optional[ConfigFormat] = None) -> 'ConfigManager':
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Detect format if not specified
        if format is None:
            if file_path.suffix.lower() in ['.json']:
                format = ConfigFormat.JSON
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            else:
                raise ValueError(f"Cannot detect format for file: {file_path}")
        
        # Load configuration data
        with open(file_path, 'r') as f:
            if format == ConfigFormat.JSON:
                data = json.load(f)
            elif format == ConfigFormat.YAML:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        # Update configuration
        self._update_config(data)
        self._config_sources.append(f"file:{file_path}")
        
        return self
    
    def load_from_dict(self, data: Dict[str, Any]) -> 'ConfigManager':
        """Load configuration from dictionary."""
        self._update_config(data)
        self._config_sources.append("dict")
        return self
    
    def load_from_env(self, prefix: str = "FIREANT_") -> 'ConfigManager':
        """Load configuration from environment variables."""
        env_config = {}
        
        # Scan environment variables
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys with underscores
                if '_' in config_key:
                    parts = config_key.split('_')
                    current = env_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Convert value to appropriate type
                    current[parts[-1]] = self._convert_env_value(value)
                else:
                    env_config[config_key] = self._convert_env_value(value)
        
        if env_config:
            self._update_config(env_config)
            self._config_sources.append(f"env:{prefix}")
        
        return self
    
    def load_from_url(self, url: str, format: ConfigFormat = ConfigFormat.JSON) -> 'ConfigManager':
        """Load configuration from URL."""
        try:
            import requests
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json() if format == ConfigFormat.JSON else yaml.safe_load(response.text)
            self._update_config(data)
            self._config_sources.append(f"url:{url}")
            
        except ImportError:
            raise ImportError("requests library is required to load configuration from URL")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from URL {url}: {e}")
        
        return self
    
    def save_to_file(self, file_path: Union[str, Path], format: ConfigFormat = ConfigFormat.JSON) -> 'ConfigManager':
        """Save current configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert configuration to dictionary
        config_dict = asdict(self._config)
        
        with open(file_path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(config_dict, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return self
    
    def get_config(self) -> FireAntConfig:
        """Get the current configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> 'ConfigManager':
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self._config.custom[key] = value
        
        return self
    
    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration."""
        return self._config.retry
    
    def get_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration."""
        return self._config.circuit_breaker
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self._config.monitoring
    
    def get_persistence_config(self) -> PersistenceConfig:
        """Get persistence configuration."""
        return self._config.persistence
    
    def get_async_config(self) -> AsyncConfig:
        """Get async configuration."""
        return self._config.async_config
    
    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get custom configuration value."""
        return self._config.custom.get(key, default)
    
    def set_custom_config(self, key: str, value: Any) -> 'ConfigManager':
        """Set custom configuration value."""
        self._config.custom[key] = value
        return self
    
    def get_config_sources(self) -> List[str]:
        """Get list of configuration sources."""
        return self._config_sources.copy()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate retry config
        if self._config.retry.max_attempts < 1:
            issues.append("retry.max_attempts must be >= 1")
        
        if self._config.retry.delay < 0:
            issues.append("retry.delay must be >= 0")
        
        if self._config.retry.backoff_factor < 1:
            issues.append("retry.backoff_factor must be >= 1")
        
        # Validate circuit breaker config
        if self._config.circuit_breaker.failure_threshold < 1:
            issues.append("circuit_breaker.failure_threshold must be >= 1")
        
        if self._config.circuit_breaker.recovery_timeout < 0:
            issues.append("circuit_breaker.recovery_timeout must be >= 0")
        
        # Validate monitoring config
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self._config.monitoring.log_level not in valid_log_levels:
            issues.append(f"monitoring.log_level must be one of {valid_log_levels}")
        
        if self._config.monitoring.metrics_history_size < 1:
            issues.append("monitoring.metrics_history_size must be >= 1")
        
        # Validate persistence config
        valid_storage_types = ["file", "memory", "database"]
        if self._config.persistence.storage_type not in valid_storage_types:
            issues.append(f"persistence.storage_type must be one of {valid_storage_types}")
        
        valid_serializers = ["json", "pickle"]
        if self._config.persistence.serializer not in valid_serializers:
            issues.append(f"persistence.serializer must be one of {valid_serializers}")
        
        if self._config.persistence.max_state_age_hours < 1:
            issues.append("persistence.max_state_age_hours must be >= 1")
        
        # Validate async config
        if self._config.async_config.max_concurrent_agents < 1:
            issues.append("async.max_concurrent_agents must be >= 1")
        
        if self._config.async_config.timeout is not None and self._config.async_config.timeout <= 0:
            issues.append("async.timeout must be > 0 if specified")
        
        return issues
    
    def _update_config(self, data: Dict[str, Any]):
        """Update configuration from dictionary data."""
        # Handle nested configuration sections
        if "retry" in data:
            retry_data = data["retry"]
            if isinstance(retry_data, dict):
                for key, value in retry_data.items():
                    if hasattr(self._config.retry, key):
                        setattr(self._config.retry, key, value)
        
        if "circuit_breaker" in data:
            cb_data = data["circuit_breaker"]
            if isinstance(cb_data, dict):
                for key, value in cb_data.items():
                    if hasattr(self._config.circuit_breaker, key):
                        setattr(self._config.circuit_breaker, key, value)
        
        if "monitoring" in data:
            mon_data = data["monitoring"]
            if isinstance(mon_data, dict):
                for key, value in mon_data.items():
                    if hasattr(self._config.monitoring, key):
                        setattr(self._config.monitoring, key, value)
        
        if "persistence" in data:
            pers_data = data["persistence"]
            if isinstance(pers_data, dict):
                for key, value in pers_data.items():
                    if hasattr(self._config.persistence, key):
                        setattr(self._config.persistence, key, value)
        
        if "async" in data or "async_config" in data:
            async_data = data.get("async") or data.get("async_config", {})
            if isinstance(async_data, dict):
                for key, value in async_data.items():
                    if hasattr(self._config.async_config, key):
                        setattr(self._config.async_config, key, value)
        
        # Handle global settings
        global_keys = ["debug", "environment", "version"]
        for key in global_keys:
            if key in data:
                setattr(self._config, key, data[key])
        
        # Handle custom configuration
        custom_keys = set(data.keys()) - {
            "retry", "circuit_breaker", "monitoring", "persistence", 
            "async", "async_config", "debug", "environment", "version"
        }
        
        for key in custom_keys:
            self._config.custom[key] = data[key]
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(sources={self._config_sources})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return f"ConfigManager(config={self._config!r}, sources={self._config_sources!r})"


# Global configuration manager instance
_default_config_manager: Optional[ConfigManager] = None


def get_default_config_manager() -> ConfigManager:
    """Get the default configuration manager."""
    global _default_config_manager
    if _default_config_manager is None:
        _default_config_manager = ConfigManager()
        
        # Try to load from environment
        _default_config_manager.load_from_env()
        
        # Try to load from common config files
        for config_file in ["fireant.json", "fireant.yaml", "fireant.yml"]:
            if os.path.exists(config_file):
                _default_config_manager.load_from_file(config_file)
                break
    
    return _default_config_manager


def set_default_config_manager(config_manager: ConfigManager):
    """Set the default configuration manager."""
    global _default_config_manager
    _default_config_manager = config_manager


def load_config(file_path: Union[str, Path], format: Optional[ConfigFormat] = None) -> ConfigManager:
    """Load configuration from file and set as default."""
    config_manager = ConfigManager()
    config_manager.load_from_file(file_path, format)
    set_default_config_manager(config_manager)
    return config_manager


def get_config() -> FireAntConfig:
    """Get the current configuration."""
    return get_default_config_manager().get_config()


# Configuration decorators
def with_config(config: Optional[FireAntConfig] = None):
    """Decorator to provide configuration to a class."""
    def decorator(cls):
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            # Inject configuration
            if config is not None:
                self._fireant_config = config
            else:
                self._fireant_config = get_config()
            
            original_init(self, *args, **kwargs)
        
        cls.__init__ = __init__
        return cls
    
    return decorator


def configure_from_file(file_path: Union[str, Path], format: Optional[ConfigFormat] = None):
    """Decorator to configure class from file."""
    def decorator(cls):
        config_manager = ConfigManager()
        config_manager.load_from_file(file_path, format)
        
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            self._fireant_config = config_manager.get_config()
            original_init(self, *args, **kwargs)
        
        cls.__init__ = __init__
        return cls
    
    return decorator
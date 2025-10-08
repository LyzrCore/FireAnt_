"""
Example demonstrating FireAnt's configuration management capabilities.
This example shows how to use configuration files, environment variables, and programmatic configuration.
"""

import os
import json
import tempfile
from pathlib import Path

from fireant import (
    Agent, AgentFlow, ConfigManager, FireAntConfig, ConfigFormat,
    RetryConfig, MonitoringConfig, PersistenceConfig,
    get_default_config_manager, set_default_config_manager, load_config,
    with_config, configure_from_file
)


class ConfigurableAgent(Agent):
    """Agent that uses configuration."""
    
    def __init__(self, name=None, config=None):
        super().__init__(name=name)
        self.config = config or get_default_config_manager().get_config()
    
    def execute(self, inputs):
        # Use configuration values
        retry_config = self.config.retry
        monitoring_config = self.config.monitoring
        
        print(f"🔧 {self.name} using config:")
        print(f"   Retry attempts: {retry_config.max_attempts}")
        print(f"   Monitoring enabled: {monitoring_config.enabled}")
        print(f"   Log level: {monitoring_config.log_level}")
        
        # Simulate work
        result = {
            "agent": self.name,
            "config_applied": True,
            "environment": self.config.environment
        }
        
        return result


def demonstrate_basic_config():
    """Demonstrate basic configuration usage."""
    print("\n=== Basic Configuration Demo ===")
    
    # Get default configuration
    config_manager = get_default_config_manager()
    config = config_manager.get_config()
    
    print(f"📋 Default configuration:")
    print(f"   Environment: {config.environment}")
    print(f"   Debug: {config.debug}")
    print(f"   Retry attempts: {config.retry.max_attempts}")
    print(f"   Monitoring enabled: {config.monitoring.enabled}")
    
    # Create agent with default config
    agent = ConfigurableAgent("BasicAgent")
    result = agent.run({"input": "test"})
    
    print(f"\n✅ Agent executed with default config")


def demonstrate_file_config():
    """Demonstrate loading configuration from file."""
    print("\n=== File Configuration Demo ===")
    
    # Create a temporary configuration file
    config_data = {
        "environment": "production",
        "debug": False,
        "retry": {
            "max_attempts": 5,
            "delay": 0.5,
            "backoff_factor": 1.5
        },
        "monitoring": {
            "enabled": True,
            "log_level": "WARNING",
            "log_file": "fireant.log"
        },
        "persistence": {
            "enabled": True,
            "storage_type": "file",
            "storage_dir": "production_states"
        },
        "custom": {
            "api_key": "prod-key-12345",
            "max_connections": 100
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_file = f.name
    
    try:
        # Load configuration from file
        config_manager = ConfigManager()
        config_manager.load_from_file(config_file)
        
        config = config_manager.get_config()
        print(f"📋 Configuration loaded from file:")
        print(f"   Environment: {config.environment}")
        print(f"   Retry attempts: {config.retry.max_attempts}")
        print(f"   Log level: {config.monitoring.log_level}")
        print(f"   Custom API key: {config.get_custom_config('api_key')}")
        
        # Create agent with file config
        agent = ConfigurableAgent("FileConfigAgent", config)
        result = agent.run({"input": "test"})
        
        print(f"✅ Agent executed with file config")
        
    finally:
        # Clean up temporary file
        os.unlink(config_file)


def demonstrate_env_config():
    """Demonstrate loading configuration from environment variables."""
    print("\n=== Environment Configuration Demo ===")
    
    # Set some environment variables
    env_vars = {
        "FIREANT_ENVIRONMENT": "staging",
        "FIREANT_DEBUG": "true",
        "FIREANT_RETRY_MAX_ATTEMPTS": "4",
        "FIREANT_MONITORING_LOG_LEVEL": "DEBUG",
        "FIREANT_PERSISTENCE_ENABLED": "true",
        "FIREANT_CUSTOM_API_ENDPOINT": "https://staging-api.example.com"
    }
    
    # Store original values
    original_vars = {}
    for key, value in env_vars.items():
        original_vars[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Load configuration from environment
        config_manager = ConfigManager()
        config_manager.load_from_env()
        
        config = config_manager.get_config()
        print(f"📋 Configuration loaded from environment:")
        print(f"   Environment: {config.environment}")
        print(f"   Debug: {config.debug}")
        print(f"   Retry attempts: {config.retry.max_attempts}")
        print(f"   Log level: {config.monitoring.log_level}")
        print(f"   Custom API endpoint: {config.get_custom_config('api_endpoint')}")
        
        # Create agent with env config
        agent = ConfigurableAgent("EnvConfigAgent", config)
        result = agent.run({"input": "test"})
        
        print(f"✅ Agent executed with environment config")
        
    finally:
        # Restore original environment variables
        for key, original_value in original_vars.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def demonstrate_programmatic_config():
    """Demonstrate programmatic configuration."""
    print("\n=== Programmatic Configuration Demo ===")
    
    # Create configuration programmatically
    config = FireAntConfig(
        environment="development",
        debug=True,
        retry=RetryConfig(max_attempts=3, delay=0.2, backoff_factor=2.0),
        monitoring=MonitoringConfig(
            enabled=True,
            log_level="INFO",
            log_file="dev.log"
        ),
        persistence=PersistenceConfig(
            enabled=True,
            storage_type="memory",
            auto_save=False
        )
    )
    
    # Add custom configuration
    config.custom.update({
        "feature_flags": {
            "new_algorithm": True,
            "experimental_ui": False
        },
        "service_urls": {
            "auth": "https://auth.dev.example.com",
            "data": "https://data.dev.example.com"
        }
    })
    
    print(f"📋 Programmatic configuration:")
    print(f"   Environment: {config.environment}")
    print(f"   Debug: {config.debug}")
    print(f"   Retry delay: {config.retry.delay}")
    print(f"   Log file: {config.monitoring.log_file}")
    print(f"   Feature flags: {config.custom['feature_flags']}")
    
    # Create agent with programmatic config
    agent = ConfigurableAgent("ProgConfigAgent", config)
    result = agent.run({"input": "test"})
    
    print(f"✅ Agent executed with programmatic config")


def demonstrate_config_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Demo ===")
    
    # Create configuration with invalid values
    config_manager = ConfigManager()
    config_manager.update_config(
        retry_max_attempts=0,  # Invalid: must be >= 1
        monitoring_log_level="INVALID",  # Invalid: not a valid log level
        persistence_max_state_age_hours=-1  # Invalid: must be >= 1
    )
    
    # Validate configuration
    issues = config_manager.validate_config()
    
    print(f"🔍 Configuration validation found {len(issues)} issues:")
    for issue in issues:
        print(f"   ❌ {issue}")
    
    # Fix the issues
    config_manager.update_config(
        retry_max_attempts=3,
        monitoring_log_level="INFO",
        persistence_max_state_age_hours=24
    )
    
    # Validate again
    issues = config_manager.validate_config()
    
    if not issues:
        print(f"✅ Configuration validation passed!")
    else:
        print(f"❌ Still has {len(issues)} issues")


def demonstrate_config_decorators():
    """Demonstrate configuration decorators."""
    print("\n=== Configuration Decorators Demo ===")
    
    # Create custom configuration
    config = FireAntConfig(
        environment="test",
        monitoring=MonitoringConfig(log_level="DEBUG")
    )
    
    # Use @with_config decorator
    @with_config(config)
    class DecoratedAgent(Agent):
        def __init__(self, name=None):
            super().__init__(name=name)
            # Configuration is injected as self._fireant_config
            self.config = self._fireant_config
        
        def execute(self, inputs):
            print(f"🔧 {self.name} using decorator config:")
            print(f"   Environment: {self.config.environment}")
            print(f"   Log level: {self.config.monitoring.log_level}")
            
            return {"decorator_config_used": True}
    
    # Create and use decorated agent
    agent = DecoratedAgent("DecoratedAgent")
    result = agent.run({"input": "test"})
    
    print(f"✅ Decorated agent executed successfully")


def demonstrate_config_sources():
    """Demonstrate multiple configuration sources."""
    print("\n=== Multiple Configuration Sources Demo ===")
    
    # Create base configuration
    base_config = {
        "environment": "development",
        "retry": {"max_attempts": 3},
        "monitoring": {"enabled": True}
    }
    
    # Create override configuration
    override_config = {
        "environment": "production",  # Override
        "debug": True,  # Add new
        "retry": {"delay": 0.5}  # Partial override
    }
    
    # Load from multiple sources
    config_manager = ConfigManager()
    config_manager.load_from_dict(base_config)
    print(f"📋 Loaded from base source: {config_manager.get_config_sources()}")
    
    config_manager.load_from_dict(override_config)
    print(f"📋 Loaded from override source: {config_manager.get_config_sources()}")
    
    config = config_manager.get_config()
    print(f"📋 Final configuration:")
    print(f"   Environment: {config.environment} (overridden)")
    print(f"   Debug: {config.debug} (added)")
    print(f"   Retry attempts: {config.retry.max_attempts} (from base)")
    print(f"   Retry delay: {config.retry.delay} (overridden)")
    print(f"   Monitoring enabled: {config.monitoring.enabled} (from base)")


def demonstrate_config_save_load():
    """Demonstrate saving and loading configuration."""
    print("\n=== Save/Load Configuration Demo ===")
    
    # Create configuration
    config_manager = ConfigManager()
    config_manager.update_config(
        environment="testing",
        debug=True,
        custom={"test_setting": "test_value"}
    )
    
    # Save to temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file = Path(temp_dir) / "config.json"
        yaml_file = Path(temp_dir) / "config.yaml"
        
        # Save as JSON
        config_manager.save_to_file(json_file, ConfigFormat.JSON)
        print(f"💾 Configuration saved to {json_file}")
        
        # Save as YAML
        config_manager.save_to_file(yaml_file, ConfigFormat.YAML)
        print(f"💾 Configuration saved to {yaml_file}")
        
        # Load from JSON
        json_config_manager = ConfigManager()
        json_config_manager.load_from_file(json_file)
        json_config = json_config_manager.get_config()
        print(f"📋 Loaded from JSON: environment={json_config.environment}")
        
        # Load from YAML
        yaml_config_manager = ConfigManager()
        yaml_config_manager.load_from_file(yaml_file)
        yaml_config = yaml_config_manager.get_config()
        print(f"📋 Loaded from YAML: environment={yaml_config.environment}")


def demonstrate_global_config():
    """Demonstrate global configuration management."""
    print("\n=== Global Configuration Demo ===")
    
    # Set global configuration
    global_config = FireAntConfig(
        environment="global",
        monitoring=MonitoringConfig(log_level="ERROR")
    )
    
    global_manager = ConfigManager(global_config)
    set_default_config_manager(global_manager)
    
    # Get configuration from different places
    config1 = get_default_config_manager().get_config()
    config2 = get_config()
    
    print(f"📋 Global configuration:")
    print(f"   Environment: {config1.environment}")
    print(f"   Log level: {config1.monitoring.log_level}")
    print(f"   Config1 == Config2: {config1.environment == config2.environment}")
    
    # Create agent that uses global config
    agent = ConfigurableAgent("GlobalConfigAgent")
    result = agent.run({"input": "test"})
    
    print(f"✅ Agent used global configuration")


if __name__ == "__main__":
    print("🔥 FireAnt Configuration Examples")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_basic_config()
    demonstrate_file_config()
    demonstrate_env_config()
    demonstrate_programmatic_config()
    demonstrate_config_validation()
    demonstrate_config_decorators()
    demonstrate_config_sources()
    demonstrate_config_save_load()
    demonstrate_global_config()
    
    print("\n✨ Configuration demo completed!")
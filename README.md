# Lyzr FireAnt 🐜🔥

A production-ready agent orchestration framework with enterprise-grade features.

FireAnt enables simple agents to work together and accomplish complex tasks through emergent collaboration.

## Why FireAnt?

FireAnt is designed for the agentic coding era—where LLM-powered platforms build applications by orchestrating specialized agents rather than generating monolithic code.

### The Agentic Advantage

**Traditional Approach:** LLM → Generate complete app → Hope it works → Debug black box

**FireAnt Approach:** LLM → Compose specialized agents → Trace execution flow → Improve incrementally

With FireAnt, you get:
- **Traceability**: See exactly which agent did what
- **Modularity**: Replace or improve individual agents without breaking the system
- **Debuggability**: Inspect the ledger at any point in the workflow
- **Composability**: Combine simple agents to create complex behaviors
- **Production-Ready**: Enterprise-grade features for real-world applications
- **Resilience**: Built-in error handling, retries, and circuit breakers
- **Observability**: Comprehensive monitoring and logging
- **Performance**: Async support for I/O-bound operations
- **Testability**: Complete testing framework
- **Flexibility**: Configuration management and state persistence

## Features

### Error Handling & Resilience
- **Retry Policies**: Configurable retry logic with exponential backoff
- **Circuit Breakers**: Prevent cascade failures with automatic recovery
- **Error Handlers**: Custom error handling and recovery strategies
- **Status Tracking**: Real-time agent status monitoring

### Monitoring & Observability
- **Built-in Logging**: Structured logging with multiple levels
- **Metrics Collection**: Performance metrics and execution tracking
- **Performance Profiling**: Detailed execution analysis
- **Custom Loggers**: Configurable logging backends

### State Persistence
- **Agent State Saving**: Save and restore agent states
- **Flow Recovery**: Resume workflows from saved states
- **Multiple Storage**: File, memory, and database storage backends
- **Serialization**: JSON and pickle serialization options

### Async Support
- **Async Agents**: Native async/await support
- **Concurrent Execution**: Run multiple agents in parallel
- **Mixed Workflows**: Combine sync and async agents
- **Performance Gains**: Better I/O-bound operation handling

### Testing Framework
- **Mock Agents**: Built-in testing utilities
- **Test Harnesses**: Comprehensive agent and flow testing
- **Assertions**: Testing helpers and validators
- **Performance Testing**: Built-in profiling tools

### Configuration Management
- **Multiple Sources**: File, environment, and programmatic config
- **Validation**: Configuration validation and error checking
- **Hot Reloading**: Runtime configuration updates
- **Environment Support**: Dev, staging, production configs

## Quick Start
```python
from fireant import Agent, AgentFlow

class DataFetcher(Agent):
    def execute(self, inputs):
        return {"data": [1, 2, 3, 4, 5]}

class DataProcessor(Agent):
    def execute(self, inputs):
        data = inputs.get("data", [])
        return {"result": sum(data)}

class ResultLogger(Agent):
    def execute(self, inputs):
        print(f"Final result: {inputs['result']}")
        return {}

# Chain agents together
flow = AgentFlow(
    start=DataFetcher()
        .next(DataProcessor())
        .next(ResultLogger())
)

# Run the workflow
flow.run(ledger={})
# Output: Final result: 15
```

## Error Handling & Resilience

```python
from fireant import Agent, AgentFlow, RetryPolicy, CircuitBreaker

class UnstableAgent(Agent):
    def __init__(self):
        super().__init__(
            retry_policy=RetryPolicy(max_attempts=3, delay=0.5, backoff_factor=2.0)
        )
    
    def execute(self, inputs):
        if random.random() < 0.7:
            raise ValueError("Random failure")
        return {"data": "success"}

# Create circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

flow = AgentFlow(
    start=UnstableAgent(),
    circuit_breaker=circuit_breaker
)

flow.run(ledger={})  # Will retry on failure
```

## Monitoring & Logging

```python
from fireant import Agent, AgentFlow, FireAntLogger, MetricsCollector

# Configure logging
logger = FireAntLogger(
    name="my_app",
    level="INFO",
    log_file="fireant.log"
)

# Enable monitoring
class MonitoredAgent(Agent):
    def __init__(self):
        super().__init__(enable_monitoring=True)
    
    def execute(self, inputs):
        return {"processed": True}

flow = AgentFlow(
    start=MonitoredAgent(),
    enable_monitoring=True
)

flow.run(ledger={})

# Get performance metrics
summary = flow.get_monitoring_summary()
print(f"Success rate: {summary['agents']['success_rate']:.2%}")
```

## State Persistence

```python
from fireant import Agent, AgentFlow, FileStateStorage, StateManager

# Configure persistence
storage = FileStateStorage(storage_dir="my_states")
state_manager = StateManager(storage)

class PersistentAgent(Agent):
    def __init__(self):
        super().__init__(enable_persistence=True, state_manager=state_manager)
    
    def execute(self, inputs):
        self.update_state("processed_items", len(inputs.get("items", [])))
        return {"result": "processed"}

flow = AgentFlow(
    start=PersistentAgent(),
    enable_persistence=True,
    state_manager=state_manager
)

# Run and save state
flow.run(ledger={"items": [1, 2, 3]})

# Resume from saved state later
flow.resume_from_state(execution_id="saved_execution_id")
```

## Async Support

```python
import asyncio
from fireant import AsyncAgent, AsyncAgentFlow, create_async_flow

class AsyncDataFetcher(AsyncAgent):
    async def execute(self, inputs):
        await asyncio.sleep(0.1)
        return {"data": [1, 2, 3, 4, 5]}

class AsyncProcessor(AsyncAgent):
    async def execute(self, inputs):
        data = inputs.get("data", [])
        processed = [x * 2 for x in data]
        return {"processed_data": processed}

# Create async flow
async_flow = create_async_flow(
    AsyncDataFetcher(),
    AsyncProcessor(),
    enable_monitoring=True
)

# Run async flow
async def main():
    result = await async_flow.run({})
    print(result)

asyncio.run(main())
```

## Testing

```python
from fireant import MockAgent, AgentTestHarness, TestSuite, assert_agent_success

# Create test harness
harness = AgentTestHarness(enable_monitoring=True)

# Test an agent
agent = MockAgent(name="TestAgent", output_data={"result": "success"})
result = harness.run_agent_test(agent, {"input": "test"})

# Assert results
assert_agent_success(result)
print(f"Agent executed in {result.execution_time:.3f}s")

# Create test suite
suite = TestSuite("MyTests")
suite.add_test(agent_test_case, test_function, "TestName")
summary = suite.run_all_tests()
```

## Configuration Management

```python
from fireant import ConfigManager, FireAntConfig, load_config

# Load from file
config_manager = load_config("fireant.json")

# Or create programmatically
config = FireAntConfig(
    environment="production",
    retry_max_attempts=5,
    monitoring_log_level="WARNING",
    persistence_enabled=True
)

# Use with agents
class ConfiguredAgent(Agent):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or get_default_config_manager().get_config()
    
    def execute(self, inputs):
        max_attempts = self.config.retry.max_attempts
        return {"attempts": max_attempts}
```

## 🏗️ Core Concepts
The Ledger Pattern
FireAnt uses a shared ledger (dictionary) that flows through your agent pipeline. Each agent reads from it, processes data, and writes back to it—accumulating context as it goes.

```python
ledger = {}
agent1.run(ledger)  # ledger now has agent1's outputs
agent2.run(ledger)  # agent2 sees agent1's results + adds its own
```

### Agent Chaining
Build sequential pipelines by chaining agents:

```python
workflow = (
    FetchAgent()
    .next(ValidateAgent())
    .next(TransformAgent())
    .next(SaveAgent())
)
```

### Task Delegation
Use ManagerAgent to delegate work to a pool of specialized workers:

```python
manager = ManagerAgent()
manager.add_worker("data_processing", DataWorker())
manager.add_worker("validation", ValidationWorker())

manager.assign_task("data_processing", {"file": "data.csv"}, priority=2)
manager.assign_task("validation", {"schema": "user"}, priority=1)

manager.process_tasks(ledger)  # Executes by priority
```

### Event-Driven Communication
Decouple agents using the event bus:

```python
class TriggerAgent(EventAgent):
    def execute(self, inputs):
        self.publish("data_ready", {"dataset": "users"})
        return {}

class ListenerAgent(Agent):
    def execute(self, inputs):
        print(f"Received: {inputs}")
        return {}

listener = ListenerAgent()
trigger = TriggerAgent()
trigger.event_bus.subscribe("data_ready", listener)

trigger.run({})  # ListenerAgent automatically runs when event fires
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fireant.git
cd fireant

# Install with pip
pip install -e .

# FireAnt has minimal dependencies - only required for advanced features
pip install fireant[all]  # Install with all optional dependencies
```

### Optional Dependencies

```bash
# For async support
pip install fireant[async]

# For persistence with database support
pip install fireant[persistence]

# For configuration with YAML support
pip install fireant[config]

# For monitoring with advanced metrics
pip install fireant[monitoring]

# For testing utilities
pip install fireant[testing]

# Install all optional dependencies
pip install fireant[all]
```
## Examples

FireAnt comes with comprehensive examples demonstrating all features:

```bash
# Run all examples
python examples/run_all_examples.py

# Individual examples
python examples/error_handling_example.py      # Error handling & retries
python examples/monitoring_example.py          # Monitoring & logging
python examples/persistence_example.py         # State persistence
python examples/async_example.py              # Async agents & flows
python examples/testing_example.py            # Testing framework
python examples/config_example.py             # Configuration management
```

### Example Highlights

- **Error Handling**: Retry policies, circuit breakers, custom error handlers
- **Monitoring**: Built-in logging, metrics collection, performance profiling
- **Persistence**: State saving, flow recovery, multiple storage backends
- **Async**: Concurrent execution, I/O-bound operations, mixed sync/async flows
- **Testing**: Mock agents, test harnesses, assertions, performance testing
- **Configuration**: File-based, environment variables, programmatic config

## Real-World Example: Building a Web Scraper
```python
class URLFetcher(Agent):
    def execute(self, inputs):
        url = inputs.get("url")
        # Fetch content logic here
        return {"html": "<html>...</html>"}

class HTMLParser(Agent):
    def execute(self, inputs):
        html = inputs.get("html", "")
        # Parse HTML logic here
        return {"data": {"title": "Example", "content": "..."}}

class DataValidator(Agent):
    def execute(self, inputs):
        data = inputs.get("data", {})
        is_valid = "title" in data and "content" in data
        return {"valid": is_valid}

class ErrorHandler(Agent):
    def execute(self, inputs):
        if not inputs.get("valid"):
            print("Error: Invalid data structure!")
        return {}

class DataSaver(Agent):
    def execute(self, inputs):
        if inputs.get("valid"):
            # Save to database
            print(f"Saved: {inputs['data']['title']}")
        return {}

# Build the workflow
scraper = AgentFlow(
    start=URLFetcher()
        .next(HTMLParser())
        .next(DataValidator())
        .next(ErrorHandler(), DataSaver())  # Parallel execution
)

scraper.run({"url": "https://example.com"})
```
## 🎯 Use Cases

LLM Agent Orchestration: Build complex AI agent systems where each agent has a specific role
Data Pipelines: Create ETL workflows with clear separation of concerns
Workflow Automation: Automate business processes with traceable, debuggable flows
Microservice Coordination: Orchestrate service calls in a clean, maintainable way
Event-Driven Systems: Build reactive applications with loose coupling

## 🧬 Philosophy
FireAnt embraces the Unix philosophy for agent systems:

Each agent does one thing well
Agents work together seamlessly
The ledger is a universal interface

Just like fire ants achieve remarkable feats through simple individual behaviors and effective collaboration, FireAnt agents combine to solve complex problems through clear composition patterns.

## 🤝 Contributing
Contributions are welcome! FireAnt is intentionally minimal, but improvements to the core patterns, bug fixes, and documentation are always appreciated.
```bash
# Fork the repo, create a branch, make your changes
git checkout -b feature/amazing-feature
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
# Open a Pull Request
```
Built for the age of LLM-powered development. 85 lines. Infinite possibilities.

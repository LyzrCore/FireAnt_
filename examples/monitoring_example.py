"""
Example demonstrating FireAnt's monitoring and logging capabilities.
This example shows how to use the built-in monitoring to track agent performance.
"""

from fireant import (
    Agent, AgentFlow, FireAntLogger, MetricsCollector, LogLevel,
    RetryPolicy, AgentStatus
)
import time
import random


class DataGenerator(Agent):
    """Generates sample data with configurable delay."""
    
    def __init__(self, data_size=10, delay=0.1):
        super().__init__(enable_monitoring=True)
        self.data_size = data_size
        self.delay = delay
    
    def execute(self, inputs):
        # Simulate work
        time.sleep(self.delay)
        
        data = list(range(1, self.data_size + 1))
        return {
            "generated_data": data,
            "count": len(data),
            "source": "DataGenerator"
        }


class DataProcessor(Agent):
    """Processes data with occasional failures for demonstration."""
    
    def __init__(self, failure_rate=0.2):
        super().__init__(
            retry_policy=RetryPolicy(
                max_attempts=3,
                delay=0.1,
                backoff_factor=1.5,
                exceptions=(ValueError,)
            ),
            enable_monitoring=True
        )
        self.failure_rate = failure_rate
    
    def execute(self, inputs):
        data = inputs.get("generated_data", [])
        
        # Simulate random failure
        if random.random() < self.failure_rate:
            raise ValueError("Random processing failure")
        
        # Simulate processing work
        time.sleep(0.2)
        
        processed = [x * 2 for x in data]
        return {
            "processed_data": processed,
            "original_count": len(data),
            "processed_count": len(processed)
        }


class DataAggregator(Agent):
    """Aggregates processed data."""
    
    def __init__(self):
        super().__init__(enable_monitoring=True)
    
    def execute(self, inputs):
        processed_data = inputs.get("processed_data", [])
        
        # Simulate aggregation work
        time.sleep(0.15)
        
        result = {
            "sum": sum(processed_data),
            "average": sum(processed_data) / len(processed_data) if processed_data else 0,
            "max": max(processed_data) if processed_data else 0,
            "min": min(processed_data) if processed_data else 0,
            "count": len(processed_data)
        }
        
        return {"aggregation_result": result}


class ResultReporter(Agent):
    """Reports final results."""
    
    def __init__(self):
        super().__init__(enable_monitoring=True)
    
    def execute(self, inputs):
        result = inputs.get("aggregation_result", {})
        
        print(f"\n📊 Final Report:")
        print(f"   Sum: {result.get('sum', 0)}")
        print(f"   Average: {result.get('average', 0):.2f}")
        print(f"   Max: {result.get('max', 0)}")
        print(f"   Min: {result.get('min', 0)}")
        print(f"   Count: {result.get('count', 0)}")
        
        return {"reported": True}


def demonstrate_basic_monitoring():
    """Demonstrate basic monitoring capabilities."""
    print("\n=== Basic Monitoring Demo ===")
    
    # Create a flow with monitoring enabled
    flow = AgentFlow(
        start=DataGenerator(data_size=5, delay=0.1)
            .next(DataProcessor(failure_rate=0.3))
            .next(DataAggregator())
            .next(ResultReporter()),
        enable_monitoring=True
    )
    
    # Run the flow multiple times to collect metrics
    for i in range(3):
        print(f"\n🔄 Running flow iteration {i+1}")
        try:
            ledger = {}
            flow.run(ledger)
        except Exception as e:
            print(f"❌ Flow failed: {e}")
    
    # Get monitoring summary
    summary = flow.get_monitoring_summary()
    print(f"\n📈 Monitoring Summary:")
    print(f"   Agent Statistics:")
    print(f"     Total agents: {summary['agents']['total']}")
    print(f"     Successful: {summary['agents']['successful']}")
    print(f"     Failed: {summary['agents']['failed']}")
    print(f"     Success rate: {summary['agents']['success_rate']:.2%}")
    print(f"     Avg execution time: {summary['agents']['avg_execution_time']:.3f}s")
    print(f"     Total retries: {summary['agents']['total_retries']}")
    
    print(f"   Flow Statistics:")
    print(f"     Total flows: {summary['flows']['total']}")
    print(f"     Successful: {summary['flows']['successful']}")
    print(f"     Success rate: {summary['flows']['success_rate']:.2%}")
    print(f"     Avg execution time: {summary['flows']['avg_execution_time']:.3f}s")


def demonstrate_custom_logging():
    """Demonstrate custom logging configuration."""
    print("\n=== Custom Logging Demo ===")
    
    # Create a custom logger
    custom_logger = FireAntLogger(
        name="custom_fireant",
        level=LogLevel.DEBUG,
        log_file="fireant_example.log"
    )
    
    # Set the custom logger for all agents
    Agent.set_logger(custom_logger)
    
    # Create a simple flow
    flow = AgentFlow(
        start=DataGenerator(data_size=3, delay=0.05)
            .next(DataProcessor(failure_rate=0.1))
            .next(DataAggregator()),
        enable_monitoring=True
    )
    
    print("📝 Running flow with custom logging (check fireant_example.log)")
    ledger = {}
    flow.run(ledger)
    
    # Get metrics collector
    metrics_collector = Agent.get_metrics_collector()
    
    # Get specific agent metrics
    processor_metrics = metrics_collector.get_agent_metrics("DataProcessor")
    print(f"\n📊 DataProcessor Metrics:")
    for metric in processor_metrics[-3:]:  # Last 3 executions
        print(f"   Status: {metric.status}, Time: {metric.execution_time:.3f}s, Retries: {metric.retry_count}")


def demonstrate_metrics_analysis():
    """Demonstrate detailed metrics analysis."""
    print("\n=== Metrics Analysis Demo ===")
    
    # Create a flow with varying performance
    flow = AgentFlow(
        start=DataGenerator(data_size=10, delay=0.2)
            .next(DataProcessor(failure_rate=0.4))
            .next(DataAggregator()),
        enable_monitoring=True
    )
    
    # Run multiple times to generate data
    print("🔄 Running flow multiple times to generate metrics...")
    for i in range(5):
        try:
            ledger = {}
            flow.run(ledger)
        except Exception as e:
            print(f"   Run {i+1} failed: {e}")
    
    # Get detailed metrics
    metrics_collector = Agent.get_metrics_collector()
    
    # Analyze agent performance
    all_agent_metrics = metrics_collector.get_agent_metrics()
    
    # Group by agent name
    agent_performance = {}
    for metric in all_agent_metrics:
        agent_name = metric.agent_name
        if agent_name not in agent_performance:
            agent_performance[agent_name] = []
        agent_performance[agent_name].append(metric)
    
    print(f"\n📊 Agent Performance Analysis:")
    for agent_name, metrics in agent_performance.items():
        total_executions = len(metrics)
        successful = sum(1 for m in metrics if m.status == "success")
        avg_time = sum(m.execution_time for m in metrics) / total_executions
        total_retries = sum(m.retry_count for m in metrics)
        
        print(f"   {agent_name}:")
        print(f"     Executions: {total_executions}")
        print(f"     Success rate: {successful/total_executions:.2%}")
        print(f"     Avg time: {avg_time:.3f}s")
        print(f"     Total retries: {total_retries}")
    
    # Flow metrics
    flow_metrics = metrics_collector.get_flow_metrics()
    print(f"\n🌊 Flow Performance:")
    for metric in flow_metrics:
        print(f"   Flow {metric.execution_id}: {metric.status}, "
              f"Time: {metric.execution_time:.3f}s, "
              f"Agents: {metric.agent_count}, "
              f"Retries: {metric.total_retries}")


if __name__ == "__main__":
    print("🔥 FireAnt Monitoring & Logging Examples")
    print("=" * 50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    demonstrate_basic_monitoring()
    demonstrate_custom_logging()
    demonstrate_metrics_analysis()
    
    print("\n✨ Monitoring demo completed!")
    print("📝 Check fireant_example.log for detailed logs")
"""
Example demonstrating FireAnt's performance metrics collection.
This example shows how to collect, analyze, and export performance metrics.
"""

import time
import random
import json
from fireant import (
    Agent, AgentFlow, PerformanceMetrics, MetricsRegistry, Timer, timer,
    MetricType, get_default_metrics, set_default_metrics
)


class MetricsEnabledAgent(Agent):
    """Agent that records performance metrics."""
    
    def __init__(self, name=None, processing_time=0.1):
        super().__init__(name=name)
        self.processing_time = processing_time
        self.metrics = get_default_metrics()
    
    def execute(self, inputs):
        # Record custom metrics
        start_time = time.time()
        
        # Simulate processing
        time.sleep(self.processing_time)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.metrics.record_custom_metric(
            f"{self.name}_processing_time",
            processing_time,
            MetricType.TIMER,
            {"agent": self.name}
        )
        
        # Record data size
        data_size = len(str(inputs))
        self.metrics.record_custom_metric(
            f"{self.name}_data_size",
            data_size,
            MetricType.GAUGE,
            {"agent": self.name}
        )
        
        return {
            "processed_by": self.name,
            "processing_time": processing_time,
            "data_size": data_size
        }


class MetricsFlow(AgentFlow):
    """Flow that records performance metrics."""
    
    def __init__(self, start=None, metrics=None):
        super().__init__(start=start)
        self.metrics = metrics or get_default_metrics()
    
    def run(self, ledger):
        with Timer(self.metrics, "flow_execution", {"flow": "MetricsFlow"}):
            return super().run(ledger)


@timer("agent_execution_timer", {"component": "example"})
def timed_agent_execution(agent, inputs):
    """Execute an agent with timing."""
    return agent.run(inputs)


def demonstrate_basic_metrics():
    """Demonstrate basic metrics collection."""
    print("\n=== Basic Metrics Demo ===")
    
    # Create metrics registry
    registry = MetricsRegistry()
    metrics = PerformanceMetrics(registry)
    
    # Record some metrics
    print("📊 Recording metrics...")
    
    # Counter metrics
    registry.counter("requests_total", 1, {"method": "GET", "status": "200"})
    registry.counter("requests_total", 1, {"method": "POST", "status": "201"})
    registry.counter("errors_total", 1, {"type": "timeout"})
    
    # Gauge metrics
    registry.gauge("active_connections", 42)
    registry.gauge("memory_usage_mb", 256.5)
    
    # Histogram metrics
    for i in range(10):
        response_time = random.uniform(0.1, 2.0)
        registry.histogram("response_time_seconds", response_time)
    
    # Timer metrics
    for i in range(5):
        duration = random.uniform(0.05, 0.5)
        registry.timer("database_query_duration", duration)
    
    # Get metrics summary
    summary = metrics.get_summary()
    print(f"📈 Metrics Summary:")
    print(f"   Uptime: {summary['uptime_seconds']:.2f}s")
    print(f"   Total metrics: {summary['total_metrics']}")
    print(f"   Total histograms: {summary['total_histograms']}")
    print(f"   Counters: {len(summary['counters'])}")
    print(f"   Gauges: {len(summary['gauges'])}")
    
    # Get timer statistics
    timer_stats = registry.get_timer_stats("database_query_duration")
    if timer_stats:
        print(f"\n⏱️ Database Query Timer Stats:")
        print(f"   Count: {timer_stats['count']}")
        print(f"   Min: {timer_stats['min']:.3f}s")
        print(f"   Max: {timer_stats['max']:.3f}s")
        print(f"   Mean: {timer_stats['mean']:.3f}s")
        print(f"   Sum: {timer_stats['sum']:.3f}s")
    
    return metrics


def demonstrate_agent_metrics():
    """Demonstrate metrics collection from agents."""
    print("\n=== Agent Metrics Demo ===")
    
    # Create metrics-enabled agents
    agent1 = MetricsEnabledAgent("DataProcessor", 0.1)
    agent2 = MetricsEnabledAgent("DataAnalyzer", 0.2)
    agent3 = MetricsEnabledAgent("DataReporter", 0.05)
    
    # Create flow with metrics
    flow = MetricsFlow(
        start=agent1.next(agent2).next(agent3)
    )
    
    # Run the flow multiple times
    print("🔄 Running flow with metrics collection...")
    for i in range(3):
        print(f"   Execution {i+1}")
        result = flow.run({"iteration": i, "data": [1, 2, 3, 4, 5]})
    
    # Get metrics summary
    metrics = get_default_metrics()
    summary = metrics.get_summary()
    
    print(f"\n📊 Agent Metrics Summary:")
    print(f"   Uptime: {summary['uptime_seconds']:.2f}s")
    print(f"   Total metrics: {summary['total_metrics']}")
    
    # Show specific agent metrics
    for agent_name in ["DataProcessor", "DataAnalyzer", "DataReporter"]:
        processing_time_metrics = metrics.registry.get_metrics(f"{agent_name}_processing_time")
        if processing_time_metrics:
            avg_time = sum(m.value for m in processing_time_metrics) / len(processing_time_metrics)
            print(f"   {agent_name} avg processing time: {avg_time:.3f}s")


def demonstrate_prometheus_export():
    """Demonstrate Prometheus metrics export."""
    print("\n=== Prometheus Export Demo ===")
    
    # Create metrics registry
    registry = MetricsRegistry()
    
    # Record various metrics
    registry.counter("http_requests_total", 100, {"method": "GET", "endpoint": "/api/data"})
    registry.counter("http_requests_total", 50, {"method": "POST", "endpoint": "/api/data"})
    registry.gauge("active_users", 1250)
    registry.histogram("request_duration_seconds", 0.123)
    registry.histogram("request_duration_seconds", 0.456)
    registry.histogram("request_duration_seconds", 0.789)
    
    # Export in Prometheus format
    prometheus_output = registry.export_prometheus()
    
    print("📄 Prometheus Export:")
    print(prometheus_output)
    
    return registry


def demonstrate_json_export():
    """Demonstrate JSON metrics export."""
    print("\n=== JSON Export Demo ===")
    
    # Get current metrics
    metrics = get_default_metrics()
    
    # Export as JSON
    json_output = metrics.export_json()
    
    print("📄 JSON Export (truncated):")
    # Parse and show summary
    data = json.loads(json_output)
    print(f"   Uptime: {data['uptime_seconds']:.2f}s")
    print(f"   Total metrics: {data['total_metrics']}")
    print(f"   Total histograms: {data['total_histograms']}")
    
    if data['metrics']:
        print(f"   Sample metrics: {len(data['metrics'])} metrics recorded")
    
    return json_output


def demonstrate_custom_metrics():
    """Demonstrate custom metrics collection."""
    print("\n=== Custom Metrics Demo ===")
    
    # Get metrics instance
    metrics = get_default_metrics()
    
    # Create custom metrics for a business process
    print("📊 Recording custom business metrics...")
    
    # Order processing metrics
    for i in range(20):
        # Simulate order processing
        order_value = random.uniform(10, 500)
        processing_time = random.uniform(0.5, 3.0)
        
        # Record custom metrics
        metrics.record_custom_metric(
            "order_value_total",
            order_value,
            MetricType.COUNTER,
            {"currency": "USD"}
        )
        
        metrics.record_custom_metric(
            "order_processing_time",
            processing_time,
            MetricType.TIMER,
            {"stage": "processing"}
        )
        
        # Record success/failure
        success = random.random() > 0.1  # 90% success rate
        metrics.record_custom_metric(
            "order_status",
            1,
            MetricType.COUNTER,
            {"status": "success" if success else "failed"}
        )
    
    # Get order processing timer stats
    order_timer_stats = metrics.registry.get_timer_stats("order_processing_time", {"stage": "processing"})
    
    if order_timer_stats:
        print(f"\n📈 Order Processing Metrics:")
        print(f"   Total orders: {order_timer_stats['count']}")
        print(f"   Avg processing time: {order_timer_stats['mean']:.2f}s")
        print(f"   Min processing time: {order_timer_stats['min']:.2f}s")
        print(f"   Max processing time: {order_timer_stats['max']:.2f}s")
    
    # Show order value counter
    order_value_counter = metrics.registry._counters.get("order_value_total{currency=USD}", 0)
    print(f"   Total order value: ${order_value_counter:.2f}")
    
    # Show order status counters
    success_counter = metrics.registry._counters.get("order_status{status=success}", 0)
    failed_counter = metrics.registry._counters.get("order_status{status=failed}", 0)
    total_orders = success_counter + failed_counter
    
    if total_orders > 0:
        success_rate = success_counter / total_orders * 100
        print(f"   Success rate: {success_rate:.1f}% ({success_counter}/{total_orders})")


def demonstrate_real_time_monitoring():
    """Demonstrate real-time metrics monitoring."""
    print("\n=== Real-time Monitoring Demo ===")
    
    # Create metrics
    metrics = get_default_metrics()
    
    # Simulate real-time metrics collection
    print("📊 Simulating real-time metrics collection...")
    
    for i in range(10):
        # Simulate some activity
        start_time = time.time()
        
        # Record metrics
        metrics.record_custom_metric(
            "real_time_events",
            1,
            MetricType.COUNTER,
            {"source": "simulator"}
        )
        
        metrics.record_custom_metric(
            "system_load",
            random.uniform(0.2, 0.8),
            MetricType.GAUGE
        )
        
        processing_time = time.time() - start_time
        metrics.record_custom_metric(
            "loop_processing_time",
            processing_time,
            MetricType.TIMER
        )
        
        # Show current status
        if i % 3 == 0:  # Show every 3rd iteration
            event_count = metrics.registry._counters.get("real_time_events{source=simulator}", 0)
            system_load = metrics.registry._gauges.get("system_load", 0)
            
            print(f"   Iteration {i+1}: Events={event_count}, Load={system_load:.2f}")
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    # Get final summary
    summary = metrics.get_summary()
    print(f"\n📈 Real-time Monitoring Summary:")
    print(f"   Total events: {summary['counters'].get('real_time_events{source=simulator}', 0)}")
    print(f"   Final system load: {summary['gauges'].get('system_load', 0):.2f}")
    
    # Get loop processing stats
    loop_stats = metrics.registry.get_timer_stats("loop_processing_time")
    if loop_stats:
        print(f"   Avg loop time: {loop_stats['mean']:.4f}s")


if __name__ == "__main__":
    print("🔥 FireAnt Performance Metrics Examples")
    print("=" * 50)
    
    # Set up default metrics
    metrics = PerformanceMetrics()
    set_default_metrics(metrics)
    
    # Run demonstrations
    demonstrate_basic_metrics()
    demonstrate_agent_metrics()
    demonstrate_prometheus_export()
    demonstrate_json_export()
    demonstrate_custom_metrics()
    demonstrate_real_time_monitoring()
    
    print("\n✨ Metrics demo completed!")
"""
Example demonstrating FireAnt's error handling and retry mechanisms.
This example shows how to build resilient agent workflows with proper error handling.
"""

from fireant import (
    Agent, AgentFlow, RetryPolicy, CircuitBreaker, AgentError, AgentStatus
)
import random
import time


class UnstableDataFetcher(Agent):
    """An agent that randomly fails to demonstrate retry mechanisms."""
    
    def __init__(self, failure_rate=0.7):
        super().__init__(
            retry_policy=RetryPolicy(
                max_attempts=3,
                delay=0.5,
                backoff_factor=2.0,
                exceptions=(ValueError,)
            )
        )
        self.failure_rate = failure_rate
    
    def execute(self, inputs):
        # Simulate random failure
        if random.random() < self.failure_rate:
            raise ValueError(f"Random failure in {self.name}")
        
        print(f"✅ {self.name} successfully fetched data")
        return {"data": [1, 2, 3, 4, 5], "source": "unstable_api"}


class DataProcessor(Agent):
    """Processes data with potential failures."""
    
    def execute(self, inputs):
        data = inputs.get("data", [])
        if not data:
            raise ValueError("No data to process")
        
        print(f"🔄 {self.name} processing {len(data)} items")
        result = sum(data)
        
        # Simulate occasional processing errors
        if random.random() < 0.2:
            raise RuntimeError("Processing error occurred")
        
        return {"result": result, "processed_count": len(data)}


class ResultLogger(Agent):
    """Logs results with custom error handling."""
    
    def __init__(self):
        def custom_error_handler(error, agent, ledger):
            print(f"🚨 Custom error handler for {agent.name}: {error}")
            ledger.setdefault("_custom_errors", []).append({
                "agent": agent.name,
                "error": str(error),
                "handled_at": time.time()
            })
        
        super().__init__(error_handler=custom_error_handler)
    
    def execute(self, inputs):
        result = inputs.get("result")
        if result is None:
            raise ValueError("No result to log")
        
        print(f"📊 Final result: {result}")
        print(f"📈 Processed {inputs.get('processed_count', 0)} items")
        return {"logged": True}


def demonstrate_retry_mechanism():
    """Demonstrate the retry mechanism with an unstable agent."""
    print("\n=== Retry Mechanism Demo ===")
    
    # Create a flow with retry-enabled agent
    flow = AgentFlow(
        start=UnstableDataFetcher(failure_rate=0.8)
            .next(DataProcessor())
            .next(ResultLogger())
    )
    
    try:
        ledger = {}
        flow.run(ledger)
        
        # Check execution history
        fetcher = flow.start
        print(f"\n📋 Execution history for {fetcher.name}:")
        for i, entry in enumerate(fetcher.execution_history):
            print(f"  {i+1}. Status: {entry['status']}, Time: {entry.get('execution_time', 0):.3f}s")
            if entry['status'] == 'retry':
                print(f"     Attempt: {entry['attempt']}, Error: {entry['error']}")
        
    except AgentError as e:
        print(f"❌ Flow failed: {e}")
        print(f"   Original error: {e.original_error}")


def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n=== Circuit Breaker Demo ===")
    
    class FailingAgent(Agent):
        def __init__(self):
            super().__init__()
            self.call_count = 0
        
        def execute(self, inputs):
            self.call_count += 1
            print(f"🔄 {self.name} call #{self.call_count}")
            raise RuntimeError("Consistent failure")
    
    # Create circuit breaker with low threshold for demo
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)
    
    flow = AgentFlow(
        start=FailingAgent(),
        circuit_breaker=circuit_breaker
    )
    
    # Try running multiple times to see circuit breaker in action
    for i in range(6):
        try:
            ledger = {}
            flow.run(ledger)
        except AgentError as e:
            print(f"❌ Attempt {i+1}: {e}")
            if "Circuit breaker is OPEN" in str(e):
                print("   ⚡ Circuit breaker is OPEN - blocking calls")
                time.sleep(2.5)  # Wait for recovery timeout
                print("   🔄 Recovery timeout passed, trying again...")


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n=== Error Handling Demo ===")
    
    def flow_error_handler(error, flow, ledger):
        print(f"🚨 Flow-level error handler: {error}")
        print(f"   Flow execution ID: {ledger.get('_flow_execution_id')}")
    
    flow = AgentFlow(
        start=UnstableDataFetcher(failure_rate=0.9)
            .next(DataProcessor())
            .next(ResultLogger()),
        error_handler=flow_error_handler
    )
    
    try:
        ledger = {}
        flow.run(ledger)
    except AgentError as e:
        print(f"\n📊 Flow execution summary:")
        summary = flow.get_execution_summary()
        print(f"   Total executions: {summary['total_executions']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Average execution time: {summary['average_execution_time']:.3f}s")
        
        # Check error ledger
        if "_errors" in ledger:
            print(f"\n📋 Errors recorded:")
            for error in ledger["_errors"]:
                print(f"   Agent: {error['agent']}, Error: {error['error']}")


if __name__ == "__main__":
    print("🔥 FireAnt Error Handling & Retry Examples")
    print("=" * 50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    demonstrate_retry_mechanism()
    demonstrate_circuit_breaker()
    demonstrate_error_handling()
    
    print("\n✨ Demo completed!")
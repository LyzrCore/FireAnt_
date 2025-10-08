"""
Example demonstrating FireAnt's testing utilities.
This example shows how to test agents and flows using the built-in testing framework.
"""

import asyncio
import time
from fireant import (
    Agent, AgentFlow, RetryPolicy,
    MockAgent, AsyncMockAgent, AgentTestHarness, AgentTestCase, TestSuite,
    assert_agent_success, assert_agent_failure, assert_output_contains,
    assert_execution_time_below, create_test_flow, create_async_test_flow,
    TestingProfiler
)


class SimpleDataProcessor(Agent):
    """Simple data processor for testing."""
    
    def __init__(self, multiplier=2):
        super().__init__()
        self.multiplier = multiplier
    
    def execute(self, inputs):
        data = inputs.get("data", [])
        
        if not data:
            raise ValueError("No data provided")
        
        processed = [x * self.multiplier for x in data]
        return {"processed_data": processed, "multiplier": self.multiplier}


class AsyncDataProcessor(AsyncMockAgent):
    """Async data processor for testing."""
    
    def __init__(self, delay=0.1):
        super().__init__(delay=delay)
    
    async def execute(self, inputs):
        result = await super().execute(inputs)
        # Add some async processing
        result["async_processed"] = True
        return result


class DataProcessorTestCase(AgentTestCase):
    """Test case for DataProcessor agent."""
    
    def __init__(self):
        super().__init__("DataProcessorTest")
        self.processor = None
    
    def setup(self):
        """Setup test case."""
        self.processor = SimpleDataProcessor(multiplier=3)
    
    def teardown(self):
        """Cleanup test case."""
        self.processor = None
    
    def test_basic_processing(self):
        """Test basic data processing."""
        inputs = {"data": [1, 2, 3, 4, 5]}
        result = self.processor.run(inputs)
        
        expected = [3, 6, 9, 12, 15]
        if result.get("processed_data") != expected:
            raise AssertionError(f"Expected {expected}, got {result.get('processed_data')}")
        
        return result
    
    def test_empty_data(self):
        """Test handling of empty data."""
        inputs = {"data": []}
        result = self.processor.run(inputs)
        
        if result.get("processed_data") != []:
            raise AssertionError("Expected empty list for empty input")
        
        return result
    
    def test_no_data_error(self):
        """Test that missing data raises an error."""
        inputs = {}
        
        try:
            self.processor.run(inputs)
            raise AssertionError("Expected ValueError for missing data")
        except ValueError:
            # Expected error
            pass
        
        return {"error_raised": True}


def demonstrate_basic_testing():
    """Demonstrate basic agent testing."""
    print("\n=== Basic Testing Demo ===")
    
    # Create test harness
    harness = AgentTestHarness(enable_monitoring=True)
    
    # Test 1: Successful agent
    print("🧪 Test 1: Successful agent")
    success_agent = MockAgent(
        name="SuccessAgent",
        output_data={"result": "success"},
        delay=0.1
    )
    
    result = harness.run_agent_test(success_agent, {"test_input": "data"})
    print(f"   Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Output: {result.output}")
    
    # Test 2: Failing agent
    print("\n🧪 Test 2: Failing agent")
    fail_agent = MockAgent(
        name="FailAgent",
        should_fail=True,
        failure_message="Intentional test failure"
    )
    
    result = harness.run_agent_test(fail_agent, {"test_input": "data"})
    print(f"   Result: {'✅ PASSED (expected failure)' if not result.success else '❌ FAILED'}")
    print(f"   Error: {result.error}")
    
    # Test 3: Agent with retry
    print("\n🧪 Test 3: Agent with retry")
    retry_agent = MockAgent(
        name="RetryAgent",
        should_fail=True,
        failure_message="Fails before retry"
    )
    retry_agent.retry_policy = RetryPolicy(max_attempts=2, delay=0.05)
    
    result = harness.run_agent_test(retry_agent, {"test_input": "data"})
    print(f"   Result: {'✅ PASSED (expected failure after retry)' if not result.success else '❌ FAILED'}")
    print(f"   Retry attempts: {len([h for h in retry_agent.execution_history if h.get('status') == 'retry'])}")


def demonstrate_flow_testing():
    """Demonstrate flow testing."""
    print("\n=== Flow Testing Demo ===")
    
    # Create test harness
    harness = AgentTestHarness(enable_monitoring=True)
    
    # Create a multi-agent flow
    flow = create_test_flow(
        MockAgent(name="Agent1", output_data={"step1": "complete"}),
        MockAgent(name="Agent2", output_data={"step2": "complete"}),
        MockAgent(name="Agent3", output_data={"step3": "complete"}),
        enable_monitoring=True
    )
    
    print("🧪 Testing multi-agent flow")
    result = harness.run_flow_test(flow, {"flow_input": "test"})
    
    print(f"   Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    
    if result.metrics:
        print(f"   Total agents: {result.metrics['agents']['total']}")
        print(f"   Success rate: {result.metrics['agents']['success_rate']:.2%}")


def demonstrate_async_testing():
    """Demonstrate async agent testing."""
    print("\n=== Async Testing Demo ===")
    
    async def run_async_tests():
        # Create test harness
        harness = AgentTestHarness(enable_monitoring=True)
        
        # Test async agent
        print("🧪 Testing async agent")
        async_agent = AsyncMockAgent(
            name="AsyncAgent",
            delay=0.1,
            output_data={"async_result": "success"}
        )
        
        result = await harness.run_async_agent_test(async_agent, {"async_input": "data"})
        
        print(f"   Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
        print(f"   Execution time: {result.execution_time:.3f}s")
        print(f"   Output: {result.output}")
        
        # Test async flow
        print("\n🧪 Testing async flow")
        async_flow = create_async_test_flow(
            AsyncMockAgent(name="AsyncAgent1", delay=0.05),
            AsyncMockAgent(name="AsyncAgent2", delay=0.05),
            enable_monitoring=True
        )
        
        flow_result = await harness.run_async_flow_test(async_flow, {"flow_input": "test"})
        
        print(f"   Result: {'✅ PASSED' if flow_result.success else '❌ FAILED'}")
        print(f"   Execution time: {flow_result.execution_time:.3f}s")
    
    # Run async tests
    asyncio.run(run_async_tests())


def demonstrate_test_suite():
    """Demonstrate test suite usage."""
    print("\n=== Test Suite Demo ===")
    
    # Create test suite
    suite = TestSuite("FireAntComponentTests")
    
    # Add individual tests
    test_case = DataProcessorTestCase()
    
    suite.add_test(test_case, test_case.test_basic_processing, "BasicProcessing")
    suite.add_test(test_case, test_case.test_empty_data, "EmptyData")
    suite.add_test(test_case, test_case.test_no_data_error, "NoDataError")
    
    # Add more tests with lambdas
    suite.add_test(
        AgentTestCase("MockAgentTest"),
        lambda: MockAgent(name="Test").run({"test": True}),
        "MockAgentBasic"
    )
    
    # Run all tests
    summary = suite.run_all_tests()
    
    return summary


def demonstrate_assertions():
    """Demonstrate testing assertions."""
    print("\n=== Testing Assertions Demo ===")
    
    harness = AgentTestHarness()
    
    # Test success assertion
    print("🧪 Testing success assertion")
    success_agent = MockAgent(name="SuccessAgent")
    result = harness.run_agent_test(success_agent)
    
    try:
        assert_agent_success(result)
        print("   ✅ Success assertion passed")
    except AssertionError as e:
        print(f"   ❌ Success assertion failed: {e}")
    
    # Test failure assertion
    print("\n🧪 Testing failure assertion")
    fail_agent = MockAgent(name="FailAgent", should_fail=True)
    fail_result = harness.run_agent_test(fail_agent)
    
    try:
        assert_agent_failure(fail_result)
        print("   ✅ Failure assertion passed")
    except AssertionError as e:
        print(f"   ❌ Failure assertion failed: {e}")
    
    # Test output assertion
    print("\n🧪 Testing output assertion")
    output_agent = MockAgent(
        name="OutputAgent",
        output_data={"key": "value", "number": 42}
    )
    output_result = harness.run_agent_test(output_agent)
    
    try:
        assert_output_contains(output_result, "key", "value")
        print("   ✅ Output assertion passed")
    except AssertionError as e:
        print(f"   ❌ Output assertion failed: {e}")
    
    # Test execution time assertion
    print("\n🧪 Testing execution time assertion")
    fast_agent = MockAgent(name="FastAgent")
    time_result = harness.run_agent_test(fast_agent)
    
    try:
        assert_execution_time_below(time_result, 1.0)
        print("   ✅ Execution time assertion passed")
    except AssertionError as e:
        print(f"   ❌ Execution time assertion failed: {e}")


def demonstrate_performance_profiling():
    """Demonstrate performance profiling."""
    print("\n=== Performance Profiling Demo ===")
    
    profiler = TestingProfiler()
    
    # Profile different operations
    with profiler.profile("Fast Operation"):
        fast_agent = MockAgent(name="FastAgent")
        fast_agent.run({"test": "data"})
    
    with profiler.profile("Slow Operation"):
        slow_agent = MockAgent(name="SlowAgent", delay=0.2)
        slow_agent.run({"test": "data"})
    
    with profiler.profile("Flow Operation"):
        flow = create_test_flow(
            MockAgent(name="Step1"),
            MockAgent(name="Step2"),
            MockAgent(name="Step3")
        )
        flow.run({"test": "data"})
    
    # Get performance summary
    summary = profiler.get_summary()
    
    print(f"📊 Performance Summary:")
    print(f"   Total profiles: {summary['total_profiles']}")
    print(f"   Total time: {summary['total_time']:.3f}s")
    print(f"   Average time: {summary['average_time']:.3f}s")
    print(f"   Max time: {summary['max_time']:.3f}s")
    print(f"   Min time: {summary['min_time']:.3f}s")
    
    print(f"\n📈 Individual Profiles:")
    for profile in summary['profiles']:
        print(f"   {profile['name']}: {profile['execution_time']:.3f}s")


def demonstrate_integration_testing():
    """Demonstrate integration testing."""
    print("\n=== Integration Testing Demo ===")
    
    # Create a realistic multi-step workflow
    class DataValidator(Agent):
        def execute(self, inputs):
            data = inputs.get("data", [])
            if not all(isinstance(x, (int, float)) for x in data):
                raise ValueError("All data must be numeric")
            return {"validated_data": data, "is_valid": True}
    
    class DataTransformer(Agent):
        def execute(self, inputs):
            data = inputs.get("validated_data", [])
            transformed = [x ** 2 for x in data]
            return {"transformed_data": transformed}
    
    class DataAggregator(Agent):
        def execute(self, inputs):
            data = inputs.get("transformed_data", [])
            result = {
                "sum": sum(data),
                "count": len(data),
                "average": sum(data) / len(data) if data else 0
            }
            return {"aggregation_result": result}
    
    # Create integration test flow
    integration_flow = create_test_flow(
        DataValidator(),
        DataTransformer(),
        DataAggregator(),
        enable_monitoring=True
    )
    
    # Test with valid data
    print("🧪 Testing with valid data")
    harness = AgentTestHarness(enable_monitoring=True)
    result = harness.run_flow_test(integration_flow, {"data": [1, 2, 3, 4, 5]})
    
    print(f"   Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
    if result.success:
        agg_result = result.output.get("aggregation_result", {})
        print(f"   Sum: {agg_result.get('sum', 0)}")
        print(f"   Average: {agg_result.get('average', 0):.2f}")
    
    # Test with invalid data
    print("\n🧪 Testing with invalid data")
    invalid_result = harness.run_flow_test(integration_flow, {"data": [1, "invalid", 3]})
    
    print(f"   Result: {'✅ PASSED (expected failure)' if not invalid_result.success else '❌ FAILED'}")
    if not invalid_result.success:
        print(f"   Expected error: {invalid_result.error}")


if __name__ == "__main__":
    print("🔥 FireAnt Testing Examples")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_basic_testing()
    demonstrate_flow_testing()
    demonstrate_async_testing()
    demonstrate_test_suite()
    demonstrate_assertions()
    demonstrate_performance_profiling()
    demonstrate_integration_testing()
    
    print("\n✨ Testing demo completed!")
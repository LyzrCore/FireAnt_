"""
FireAnt testing utilities.
Provides testing helpers and utilities for agents and flows.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
from unittest.mock import Mock, patch
from contextlib import contextmanager
from dataclasses import dataclass
import tempfile
import os

from .core import Agent, AgentFlow, AgentStatus
from .async_core import AsyncAgent, AsyncAgentFlow
from .monitoring import MetricsCollector, FireAntLogger
from .persistence import MemoryStateStorage, StateManager


@dataclass
class TestResult:
    """Result of a test execution."""
    success: bool
    execution_time: float
    output: Dict[str, Any]
    error: Optional[Exception] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None


class MockAgent(Agent):
    """Mock agent for testing purposes."""
    
    def __init__(self, name=None, output_data=None, delay=0, should_fail=False, 
                 failure_message="Mock agent failed", enable_monitoring=False):
        super().__init__(name=name, enable_monitoring=enable_monitoring)
        self.output_data = output_data or {"mock_output": True}
        self.delay = delay
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.execute_count = 0
    
    def execute(self, inputs):
        self.execute_count += 1
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        
        # Include input data in output for testing
        output = self.output_data.copy()
        output["input_received"] = bool(inputs)
        output["execute_count"] = self.execute_count
        
        return output


class AsyncMockAgent(AsyncAgent):
    """Async mock agent for testing purposes."""
    
    def __init__(self, name=None, output_data=None, delay=0, should_fail=False,
                 failure_message="Async mock agent failed", enable_monitoring=False):
        super().__init__(name=name, enable_monitoring=enable_monitoring)
        self.output_data = output_data or {"async_mock_output": True}
        self.delay = delay
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.execute_count = 0
    
    async def execute(self, inputs):
        self.execute_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        
        # Include input data in output for testing
        output = self.output_data.copy()
        output["input_received"] = bool(inputs)
        output["execute_count"] = self.execute_count
        
        return output


class AgentTestHarness:
    """Test harness for agents and flows."""
    
    def __init__(self, enable_monitoring=False, enable_persistence=False):
        self.enable_monitoring = enable_monitoring
        self.enable_persistence = enable_persistence
        self.metrics_collector = None
        self.logger = None
        self.state_manager = None
        
        if enable_monitoring:
            self.metrics_collector = MetricsCollector()
            self.logger = FireAntLogger()
        
        if enable_persistence:
            storage = MemoryStateStorage()
            self.state_manager = StateManager(storage)
    
    def run_agent_test(self, agent: Agent, inputs: Dict[str, Any] = None) -> TestResult:
        """Run a single agent test."""
        if inputs is None:
            inputs = {}
        
        start_time = time.time()
        
        try:
            # Configure agent if needed
            if self.enable_monitoring and hasattr(agent, 'enable_monitoring'):
                agent.enable_monitoring = True
            
            if self.enable_persistence and hasattr(agent, 'enable_persistence'):
                agent.enable_persistence = True
                agent.state_manager = self.state_manager
            
            # Run the agent
            result = agent.run(inputs.copy())
            execution_time = time.time() - start_time
            
            # Collect metrics if enabled
            metrics = None
            if self.metrics_collector:
                metrics = self.metrics_collector.get_performance_summary()
            
            return TestResult(
                success=True,
                execution_time=execution_time,
                output=result,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                success=False,
                execution_time=execution_time,
                output={},
                error=e,
                metrics=self.metrics_collector.get_performance_summary() if self.metrics_collector else None
            )
    
    async def run_async_agent_test(self, agent: AsyncAgent, inputs: Dict[str, Any] = None) -> TestResult:
        """Run a single async agent test."""
        if inputs is None:
            inputs = {}
        
        start_time = time.time()
        
        try:
            # Configure agent if needed
            if self.enable_monitoring and hasattr(agent, 'enable_monitoring'):
                agent.enable_monitoring = True
            
            if self.enable_persistence and hasattr(agent, 'enable_persistence'):
                agent.enable_persistence = True
                agent.state_manager = self.state_manager
            
            # Run the agent
            result = await agent.run(inputs.copy())
            execution_time = time.time() - start_time
            
            # Collect metrics if enabled
            metrics = None
            if self.metrics_collector:
                metrics = self.metrics_collector.get_performance_summary()
            
            return TestResult(
                success=True,
                execution_time=execution_time,
                output=result,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                success=False,
                execution_time=execution_time,
                output={},
                error=e,
                metrics=self.metrics_collector.get_performance_summary() if self.metrics_collector else None
            )
    
    def run_flow_test(self, flow: AgentFlow, inputs: Dict[str, Any] = None) -> TestResult:
        """Run a flow test."""
        if inputs is None:
            inputs = {}
        
        start_time = time.time()
        
        try:
            # Configure flow if needed
            if self.enable_monitoring and hasattr(flow, 'enable_monitoring'):
                flow.enable_monitoring = True
            
            if self.enable_persistence and hasattr(flow, 'enable_persistence'):
                flow.enable_persistence = True
                flow.state_manager = self.state_manager
            
            # Run the flow
            result = flow.run(inputs.copy())
            execution_time = time.time() - start_time
            
            # Collect metrics if enabled
            metrics = None
            if self.enable_monitoring:
                metrics = flow.get_monitoring_summary()
            
            return TestResult(
                success=True,
                execution_time=execution_time,
                output=result,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                success=False,
                execution_time=execution_time,
                output={},
                error=e,
                metrics=flow.get_monitoring_summary() if self.enable_monitoring else None
            )
    
    async def run_async_flow_test(self, flow: AsyncAgentFlow, inputs: Dict[str, Any] = None) -> TestResult:
        """Run an async flow test."""
        if inputs is None:
            inputs = {}
        
        start_time = time.time()
        
        try:
            # Configure flow if needed
            if self.enable_monitoring and hasattr(flow, 'enable_monitoring'):
                flow.enable_monitoring = True
            
            if self.enable_persistence and hasattr(flow, 'enable_persistence'):
                flow.enable_persistence = True
                flow.state_manager = self.state_manager
            
            # Run the flow
            result = await flow.run(inputs.copy())
            execution_time = time.time() - start_time
            
            # Collect metrics if enabled
            metrics = None
            if self.enable_monitoring:
                metrics = flow.get_monitoring_summary()
            
            return TestResult(
                success=True,
                execution_time=execution_time,
                output=result,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                success=False,
                execution_time=execution_time,
                output={},
                error=e,
                metrics=flow.get_monitoring_summary() if self.enable_monitoring else None
            )


class AgentTestCase:
    """Base class for agent test cases."""
    
    def __init__(self, name: str):
        self.name = name
        self.setup_complete = False
        self.teardown_complete = False
    
    def setup(self):
        """Setup method called before each test."""
        pass
    
    def teardown(self):
        """Teardown method called after each test."""
        pass
    
    def run_test(self, test_func: Callable) -> TestResult:
        """Run a test function with setup and teardown."""
        try:
            self.setup()
            self.setup_complete = True
            
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time
            
            return TestResult(
                success=True,
                execution_time=execution_time,
                output=result if isinstance(result, dict) else {"result": result}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                execution_time=execution_time,
                output={},
                error=e
            )
        
        finally:
            try:
                self.teardown()
                self.teardown_complete = True
            except Exception as e:
                print(f"Teardown error in {self.name}: {e}")


class TestSuite:
    """Test suite for running multiple tests."""
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases = []
        self.results = []
    
    def add_test(self, test_case: AgentTestCase, test_func: Callable, test_name: str = None):
        """Add a test to the suite."""
        if test_name is None:
            test_name = f"{test_case.name}_{len(self.test_cases)}"
        
        self.test_cases.append({
            "name": test_name,
            "test_case": test_case,
            "test_func": test_func
        })
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests in the suite."""
        self.results = []
        
        print(f"🧪 Running test suite: {self.name}")
        print(f"📊 Total tests: {len(self.test_cases)}")
        
        passed = 0
        failed = 0
        total_time = 0
        
        for i, test in enumerate(self.test_cases):
            print(f"\n🔄 Running test {i+1}/{len(self.test_cases)}: {test['name']}")
            
            result = test["test_case"].run_test(test["test_func"])
            result.test_name = test["name"]
            
            self.results.append(result)
            total_time += result.execution_time
            
            if result.success:
                print(f"✅ {test['name']}: PASSED ({result.execution_time:.3f}s)")
                passed += 1
            else:
                print(f"❌ {test['name']}: FAILED ({result.execution_time:.3f}s)")
                print(f"   Error: {result.error}")
                failed += 1
        
        success_rate = passed / len(self.test_cases) if self.test_cases else 0
        
        summary = {
            "suite_name": self.name,
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "total_time": total_time,
            "average_time": total_time / len(self.test_cases) if self.test_cases else 0,
            "results": self.results
        }
        
        print(f"\n📊 Test Suite Summary:")
        print(f"   Total: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.2%}")
        print(f"   Total Time: {summary['total_time']:.3f}s")
        print(f"   Average Time: {summary['average_time']:.3f}s")
        
        return summary


@contextmanager
def temp_directory():
    """Context manager for temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def assert_agent_success(result: TestResult):
    """Assert that an agent test result is successful."""
    if not result.success:
        raise AssertionError(f"Agent test failed: {result.error}")
    return result


def assert_agent_failure(result: TestResult, expected_error_type: type = None):
    """Assert that an agent test result failed as expected."""
    if result.success:
        raise AssertionError("Expected agent test to fail, but it succeeded")
    
    if expected_error_type and not isinstance(result.error, expected_error_type):
        raise AssertionError(f"Expected error type {expected_error_type}, got {type(result.error)}")
    
    return result


def assert_output_contains(result: TestResult, key: str, value: Any = None):
    """Assert that the output contains a specific key (and optionally value)."""
    if key not in result.output:
        raise AssertionError(f"Output does not contain key: {key}")
    
    if value is not None and result.output[key] != value:
        raise AssertionError(f"Expected {key}={value}, got {result.output[key]}")
    
    return result


def assert_execution_time_below(result: TestResult, max_time: float):
    """Assert that execution time is below a threshold."""
    if result.execution_time > max_time:
        raise AssertionError(f"Execution time {result.execution_time:.3f}s exceeds maximum {max_time:.3f}s")
    
    return result


def create_test_flow(*agents, **kwargs) -> AgentFlow:
    """Create a test flow with the given agents."""
    if not agents:
        raise ValueError("At least one agent is required")
    
    # Chain agents together
    start_agent = agents[0]
    for agent in agents[1:]:
        start_agent.next(agent)
    
    return AgentFlow(start=start_agent, **kwargs)


def create_async_test_flow(*agents, **kwargs) -> AsyncAgentFlow:
    """Create an async test flow with the given agents."""
    if not agents:
        raise ValueError("At least one agent is required")
    
    # Chain agents together
    start_agent = agents[0]
    for agent in agents[1:]:
        start_agent.next(agent)
    
    return AsyncAgentFlow(start=start_agent, **kwargs)


# Performance testing utilities
class PerformanceProfiler:
    """Performance profiler for agents and flows."""
    
    def __init__(self):
        self.profiles = []
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile = {
                "name": name,
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "timestamp": time.time()
            }
            
            self.profiles.append(profile)
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            return 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.profiles:
            return {"message": "No profiles recorded"}
        
        total_time = sum(p["execution_time"] for p in self.profiles)
        avg_time = total_time / len(self.profiles)
        max_time = max(p["execution_time"] for p in self.profiles)
        min_time = min(p["execution_time"] for p in self.profiles)
        
        return {
            "total_profiles": len(self.profiles),
            "total_time": total_time,
            "average_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "profiles": self.profiles
        }
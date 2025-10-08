"""
FireAnt async core module.
Provides async versions of Agent and AgentFlow for better performance with I/O-bound operations.
"""

import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from .monitoring import MonitoringMixin, FireAntLogger, MetricsCollector, AgentMetric, FlowMetric, PerformanceProfiler
from .persistence import StateManager, get_default_state_manager


class AsyncAgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class AsyncAgentError(Exception):
    def __init__(self, message: str, agent_name: str, original_error: Exception = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.original_error = original_error


class AsyncRetryPolicy:
    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0,
                 exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions


class AsyncCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise AsyncAgentError("Circuit breaker is OPEN", "AsyncCircuitBreaker")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e


class AsyncAgent(MonitoringMixin):
    def __init__(self, name=None, retry_policy: Optional[AsyncRetryPolicy] = None,
                 error_handler: Optional[Callable] = None, enable_monitoring: bool = True,
                 enable_persistence: bool = False, state_manager: Optional[StateManager] = None):
        self.name = name or self.__class__.__name__
        self._next = []
        self._event_bus = None
        self.retry_policy = retry_policy
        self.error_handler = error_handler
        self.status = AsyncAgentStatus.PENDING
        self.execution_history = []
        self.enable_monitoring = enable_monitoring
        self.enable_persistence = enable_persistence
        self.state_manager = state_manager or get_default_state_manager()
        self._custom_state = {}

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async execute method - must be implemented by subclasses."""
        raise NotImplementedError

    def prepare(self, ledger: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare method - kept synchronous for simplicity."""
        return ledger

    def next(self, *agents):
        """Chain agents together."""
        self._next.extend(agents)
        return self

    async def run(self, ledger: Dict[str, Any]):
        """Async run method."""
        execution_id = self._get_execution_id()
        self.status = AsyncAgentStatus.RUNNING
        start_time = time.time()
        
        # Monitoring setup
        if self.enable_monitoring:
            logger = self.get_logger()
            metrics_collector = self.get_metrics_collector()
            
            # Log agent start
            inputs = self.prepare(ledger)
            logger.log_agent_start(self.name, execution_id, inputs)
            
            # Start performance profiling
            with PerformanceProfiler(self.name, execution_id, logger, metrics_collector):
                try:
                    if self.retry_policy:
                        result = await self._run_with_retry(ledger, execution_id)
                    else:
                        result = await self._run_once(ledger, execution_id)
                    
                    self.status = AsyncAgentStatus.SUCCESS
                    execution_time = time.time() - start_time
                    
                    # Record metric
                    outputs = {k: v for k, v in ledger.items() if k not in inputs}
                    metric = AgentMetric(
                        agent_name=self.name,
                        execution_id=execution_id,
                        status="success",
                        start_time=start_time,
                        end_time=time.time(),
                        execution_time=execution_time,
                        input_size=self._calculate_data_size(inputs),
                        output_size=self._calculate_data_size(outputs),
                        retry_count=len([h for h in self.execution_history if h.get("status") == "retry"])
                    )
                    metrics_collector.record_agent_metric(metric)
                    
                    # Log success
                    logger.log_agent_success(self.name, execution_id, execution_time, outputs)
                    
                    # Save state if persistence is enabled
                    if self.enable_persistence:
                        self.save_state(ledger, execution_id)
                    
                    # Continue to next agents
                    for agent in self._next:
                        if isinstance(agent, AsyncAgent):
                            await agent.run(ledger)
                        else:
                            # Fallback to sync agent
                            agent.run(ledger)
                    
                    return ledger
                    
                except Exception as e:
                    self.status = AsyncAgentStatus.FAILED
                    execution_time = time.time() - start_time
                    
                    # Record metric
                    metric = AgentMetric(
                        agent_name=self.name,
                        execution_id=execution_id,
                        status="failed",
                        start_time=start_time,
                        end_time=time.time(),
                        execution_time=execution_time,
                        input_size=self._calculate_data_size(inputs),
                        output_size=0,
                        error_message=str(e),
                        retry_count=len([h for h in self.execution_history if h.get("status") == "retry"])
                    )
                    metrics_collector.record_agent_metric(metric)
                    
                    # Log failure
                    logger.log_agent_failure(self.name, execution_id, execution_time, e)
                    
                    if self.error_handler:
                        if asyncio.iscoroutinefunction(self.error_handler):
                            await self.error_handler(e, self, ledger)
                        else:
                            self.error_handler(e, self, ledger)
                    else:
                        # Default error handling
                        error_info = {
                            "error": str(e),
                            "agent": self.name,
                            "traceback": traceback.format_exc()
                        }
                        ledger.setdefault("_errors", []).append(error_info)
                    
                    # Re-raise the exception to stop the flow unless handled
                    raise AsyncAgentError(f"Agent {self.name} failed: {str(e)}", self.name, e)
        else:
            # Original behavior without monitoring
            try:
                if self.retry_policy:
                    result = await self._run_with_retry(ledger)
                else:
                    result = await self._run_once(ledger)
                
                self.status = AsyncAgentStatus.SUCCESS
                execution_time = time.time() - start_time
                self.execution_history.append({
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": time.time()
                })
                
                # Continue to next agents
                for agent in self._next:
                    if isinstance(agent, AsyncAgent):
                        await agent.run(ledger)
                    else:
                        agent.run(ledger)
                
                return ledger
                
            except Exception as e:
                self.status = AsyncAgentStatus.FAILED
                execution_time = time.time() - start_time
                self.execution_history.append({
                    "status": "failed",
                    "error": str(e),
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "traceback": traceback.format_exc()
                })
                
                if self.error_handler:
                    if asyncio.iscoroutinefunction(self.error_handler):
                        await self.error_handler(e, self, ledger)
                    else:
                        self.error_handler(e, self, ledger)
                else:
                    # Default error handling
                    error_info = {
                        "error": str(e),
                        "agent": self.name,
                        "traceback": traceback.format_exc()
                    }
                    ledger.setdefault("_errors", []).append(error_info)
                
                # Re-raise the exception to stop the flow unless handled
                raise AsyncAgentError(f"Agent {self.name} failed: {str(e)}", self.name, e)

    async def _run_once(self, ledger: Dict[str, Any], execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Run agent once without retry."""
        inputs = self.prepare(ledger)
        outputs = await self.execute(inputs)
        ledger.update(outputs or {})
        return ledger

    async def _run_with_retry(self, ledger: Dict[str, Any], execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Run agent with retry logic."""
        last_exception = None
        logger = self.get_logger() if self.enable_monitoring and execution_id else None
        
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                if attempt > 1:
                    self.status = AsyncAgentStatus.RETRYING
                    delay = self.retry_policy.delay * (self.retry_policy.backoff_factor ** (attempt - 2))
                    await asyncio.sleep(delay)
                
                return await self._run_once(ledger, execution_id)
                
            except self.retry_policy.exceptions as e:
                last_exception = e
                if attempt == self.retry_policy.max_attempts:
                    break
                
                # Log retry attempt
                if logger:
                    logger.log_agent_retry(self.name, execution_id, attempt, e)
                
                self.execution_history.append({
                    "status": "retry",
                    "attempt": attempt,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        raise last_exception

    def get_state(self) -> Dict[str, Any]:
        """Get agent's custom state for persistence."""
        return self._custom_state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """Set agent's custom state from persistence."""
        self._custom_state.update(state)
    
    def update_state(self, key: str, value: Any):
        """Update a specific value in agent's custom state."""
        self._custom_state[key] = value
    
    def save_state(self, ledger: Dict[str, Any], execution_id: Optional[str] = None) -> bool:
        """Save agent state to persistent storage."""
        if not self.enable_persistence:
            return True
        
        return self.state_manager.save_agent_state(self, ledger, execution_id)
    
    def load_state(self, execution_id: str) -> bool:
        """Load agent state from persistent storage."""
        if not self.enable_persistence:
            return False
        
        agent_state = self.state_manager.load_agent_state(self.name, execution_id)
        if agent_state:
            self.set_state(agent_state.custom_state)
            return True
        return False

    @property
    def event_bus(self):
        if not self._event_bus:
            from .core import EventBus
            self._event_bus = EventBus()
        return self._event_bus


class AsyncAgentFlow(MonitoringMixin):
    def __init__(self, start=None, error_handler: Optional[Callable] = None,
                 circuit_breaker: Optional[AsyncCircuitBreaker] = None, enable_monitoring: bool = True,
                 enable_persistence: bool = False, state_manager: Optional[StateManager] = None):
        self.start = start
        self.agents = set()
        self.error_handler = error_handler
        self.circuit_breaker = circuit_breaker
        self.execution_history = []
        self.enable_monitoring = enable_monitoring
        self.enable_persistence = enable_persistence
        self.state_manager = state_manager or get_default_state_manager()

    async def run(self, ledger: Dict[str, Any]):
        """Async run method for the flow."""
        execution_id = str(int(time.time() * 1000))
        ledger.setdefault("_flow_execution_id", execution_id)
        start_time = time.time()
        
        # Monitoring setup
        if self.enable_monitoring:
            logger = self.get_logger()
            metrics_collector = self.get_metrics_collector()
            
            # Log flow start
            flow_id = f"flow_{execution_id}"
            logger.log_flow_start(flow_id, execution_id)
            
            # Start performance profiling
            with PerformanceProfiler(flow_id, execution_id, logger, metrics_collector):
                try:
                    if self.circuit_breaker:
                        result = await self.circuit_breaker.call(self._execute_flow, ledger)
                    else:
                        result = await self._execute_flow(ledger)
                    
                    execution_time = time.time() - start_time
                    
                    # Count agents and their status
                    agent_count = self._count_agents()
                    successful_agents = self._count_successful_agents()
                    failed_agents = agent_count - successful_agents
                    total_retries = self._count_total_retries()
                    
                    # Record flow metric
                    from .monitoring import FlowMetric
                    metric = FlowMetric(
                        flow_id=flow_id,
                        execution_id=execution_id,
                        status="success",
                        start_time=start_time,
                        end_time=time.time(),
                        execution_time=execution_time,
                        agent_count=agent_count,
                        successful_agents=successful_agents,
                        failed_agents=failed_agents,
                        total_retries=total_retries
                    )
                    metrics_collector.record_flow_metric(metric)
                    
                    # Log success
                    logger.log_flow_success(flow_id, execution_id, execution_time)
                    
                    # Save flow state if persistence is enabled
                    if self.enable_persistence:
                        self.state_manager.save_flow_state(self, ledger, execution_id)
                    
                    # Update execution history
                    self.execution_history.append({
                        "status": "success",
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                        "flow_id": execution_id
                    })
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Record flow metric
                    from .monitoring import FlowMetric
                    metric = FlowMetric(
                        flow_id=flow_id,
                        execution_id=execution_id,
                        status="failed",
                        start_time=start_time,
                        end_time=time.time(),
                        execution_time=execution_time,
                        agent_count=self._count_agents(),
                        successful_agents=0,
                        failed_agents=self._count_agents(),
                        total_retries=0,
                        error_message=str(e)
                    )
                    metrics_collector.record_flow_metric(metric)
                    
                    # Log failure
                    logger.log_flow_failure(flow_id, execution_id, execution_time, e)
                    
                    if self.error_handler:
                        if asyncio.iscoroutinefunction(self.error_handler):
                            await self.error_handler(e, self, ledger)
                        else:
                            self.error_handler(e, self, ledger)
                    else:
                        # Default error handling
                        error_info = {
                            "error": str(e),
                            "flow": "AsyncAgentFlow",
                            "traceback": traceback.format_exc(),
                            "flow_id": execution_id
                        }
                        ledger.setdefault("_errors", []).append(error_info)
                    
                    # Update execution history
                    self.execution_history.append({
                        "status": "failed",
                        "error": str(e),
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                        "flow_id": execution_id,
                        "traceback": traceback.format_exc()
                    })
                    
                    raise e
        else:
            # Original behavior without monitoring
            try:
                if self.circuit_breaker:
                    result = await self.circuit_breaker.call(self._execute_flow, ledger)
                else:
                    result = await self._execute_flow(ledger)
                
                execution_time = time.time() - start_time
                self.execution_history.append({
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "flow_id": ledger["_flow_execution_id"]
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.execution_history.append({
                    "status": "failed",
                    "error": str(e),
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "flow_id": ledger["_flow_execution_id"],
                    "traceback": traceback.format_exc()
                })
                
                if self.error_handler:
                    if asyncio.iscoroutinefunction(self.error_handler):
                        await self.error_handler(e, self, ledger)
                    else:
                        self.error_handler(e, self, ledger)
                
                raise e

    async def _execute_flow(self, ledger: Dict[str, Any]):
        """Execute the flow."""
        if self.start:
            if isinstance(self.start, AsyncAgent):
                return await self.start.run(ledger)
            else:
                return self.start.run(ledger)
        return ledger

    def add_parallel_branch(self, agents: List[AsyncAgent]):
        """Add parallel branch of agents."""
        self.agents.update(agents)
        return self

    def add_conditional_branch(self, name: str, agent: AsyncAgent):
        """Add conditional branch."""
        setattr(self, f"cond_{name}", agent)
        return self

    def add_checkpoint(self, name: str, cond: Callable[[Dict[str, Any]], bool]):
        """Add checkpoint."""
        setattr(self, f"chk_{name}", cond)
        return self

    def _count_agents(self) -> int:
        """Count the total number of agents in this flow."""
        if not self.start:
            return 0
        
        visited = set()
        queue = [self.start]
        count = 0
        
        while queue:
            agent = queue.pop(0)
            if agent not in visited:
                visited.add(agent)
                count += 1
                queue.extend(agent._next)
        
        return count
    
    def _count_successful_agents(self) -> int:
        """Count agents that have successful status."""
        if not self.start:
            return 0
        
        visited = set()
        queue = [self.start]
        count = 0
        
        while queue:
            agent = queue.pop(0)
            if agent not in visited:
                visited.add(agent)
                if hasattr(agent, 'status') and agent.status == AsyncAgentStatus.SUCCESS:
                    count += 1
                queue.extend(agent._next)
        
        return count
    
    def _count_total_retries(self) -> int:
        """Count total retry attempts across all agents."""
        if not self.start:
            return 0
        
        visited = set()
        queue = [self.start]
        total_retries = 0
        
        while queue:
            agent = queue.pop(0)
            if agent not in visited:
                visited.add(agent)
                if hasattr(agent, 'execution_history'):
                    total_retries += sum(1 for h in agent.execution_history if h.get("status") == "retry")
                queue.extend(agent._next)
        
        return total_retries

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        if not self.execution_history:
            return {"message": "No executions recorded"}
        
        latest = self.execution_history[-1]
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h["status"] == "success")
        
        return {
            "latest_execution": latest,
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time": sum(h.get("execution_time", 0) for h in self.execution_history) / total_executions
        }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary including metrics."""
        if not self.enable_monitoring:
            return {"message": "Monitoring is disabled"}
        
        metrics_collector = self.get_metrics_collector()
        return metrics_collector.get_performance_summary()

    def save_flow_state(self, ledger: Dict[str, Any], execution_id: str) -> bool:
        """Save flow state to persistent storage."""
        if not self.enable_persistence:
            return True
        
        return self.state_manager.save_flow_state(self, ledger, execution_id)
    
    def load_flow_state(self, execution_id: str) -> bool:
        """Load flow state from persistent storage."""
        if not self.enable_persistence:
            return False
        
        flow_state = self.state_manager.load_flow_state(execution_id)
        if flow_state:
            # Restore agent states
            for agent_state in flow_state.agent_states:
                # Find the corresponding agent and restore its state
                agent = self._find_agent_by_name(agent_state.agent_name)
                if agent:
                    agent.set_state(agent_state.custom_state)
            return True
        return False
    
    def _find_agent_by_name(self, name: str) -> Optional[AsyncAgent]:
        """Find an agent in the flow by name."""
        if not self.start:
            return None
        
        visited = set()
        queue = [self.start]
        
        while queue:
            agent = queue.pop(0)
            if agent not in visited:
                visited.add(agent)
                if agent.name == name:
                    return agent
                queue.extend(agent._next)
        
        return None
    
    async def resume_from_state(self, execution_id: str, ledger: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resume flow execution from saved state."""
        if not self.load_flow_state(execution_id):
            raise ValueError(f"Could not load flow state for execution ID: {execution_id}")
        
        # Use provided ledger or create new one
        if ledger is None:
            ledger = {}
        
        # Set execution ID in ledger
        ledger["_flow_execution_id"] = execution_id
        
        # Resume execution
        return await self.run(ledger)


# Utility functions for running async flows
async def run_async_flow(flow: AsyncAgentFlow, ledger: Dict[str, Any] = None) -> Dict[str, Any]:
    """Utility function to run an async flow."""
    if ledger is None:
        ledger = {}
    return await flow.run(ledger)


def create_async_flow(*agents, **kwargs) -> AsyncAgentFlow:
    """Utility function to create an async flow from agents."""
    if not agents:
        raise ValueError("At least one agent is required")
    
    # Chain agents together
    start_agent = agents[0]
    for agent in agents[1:]:
        start_agent.next(agent)
    
    return AsyncAgentFlow(start=start_agent, **kwargs)
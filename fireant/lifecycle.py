"""
FireAnt agent lifecycle management.
Provides comprehensive lifecycle management for agents and flows.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio

from .core import Agent, AgentFlow, AgentStatus
from .async_core import AsyncAgent, AsyncAgentFlow, AsyncAgentStatus
from .monitoring import FireAntLogger, get_default_logger
from .metrics import PerformanceMetrics, get_default_metrics


class LifecycleEvent(Enum):
    """Lifecycle events."""
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    RESUMED = "resumed"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class LifecycleState(Enum):
    """Lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class LifecycleEventInfo:
    """Information about a lifecycle event."""
    event: LifecycleEvent
    timestamp: float
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


class LifecycleListener(ABC):
    """Abstract base class for lifecycle listeners."""
    
    @abstractmethod
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Called when a lifecycle event occurs."""
        pass


class LifecycleManager:
    """Manages the lifecycle of agents and flows."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = LifecycleState.UNINITIALIZED
        self.listeners: List[LifecycleListener] = []
        self.event_history: List[LifecycleEventInfo] = []
        self._lock = threading.Lock()
        self._logger = get_default_logger()
        self._metrics = get_default_metrics()
    
    def add_listener(self, listener: LifecycleListener):
        """Add a lifecycle listener."""
        with self._lock:
            self.listeners.append(listener)
    
    def remove_listener(self, listener: LifecycleListener):
        """Remove a lifecycle listener."""
        with self._lock:
            if listener in self.listeners:
                self.listeners.remove(listener)
    
    def _notify_listeners(self, event: LifecycleEvent, source: str, 
                          data: Dict[str, Any] = None, error: Exception = None):
        """Notify all listeners of a lifecycle event."""
        event_info = LifecycleEventInfo(
            event=event,
            timestamp=time.time(),
            source=source,
            data=data or {},
            error=error
        )
        
        with self._lock:
            self.event_history.append(event_info)
            
            # Record metrics
            self._metrics.record_custom_metric(
                f"lifecycle_event_{event.value}",
                1,
                labels={"source": source, "state": self.state.value}
            )
            
            # Log event
            self._logger.info(
                f"Lifecycle event: {event.value} for {source} (state: {self.state.value})"
            )
            
            # Notify listeners
            for listener in self.listeners:
                try:
                    listener.on_lifecycle_event(event_info)
                except Exception as e:
                    self._logger.error(f"Error in lifecycle listener: {e}")
    
    def _set_state(self, state: LifecycleState, source: str = None):
        """Set the current state."""
        old_state = self.state
        self.state = state
        
        self._logger.info(f"State transition: {old_state.value} -> {state.value} for {source or self.name}")
    
    def initialize(self, source: str = None):
        """Initialize the component."""
        source = source or self.name
        
        with self._lock:
            if self.state not in [LifecycleState.UNINITIALIZED, LifecycleState.TERMINATED]:
                raise RuntimeError(f"Cannot initialize in state: {self.state.value}")
            
            self._notify_listeners(LifecycleEvent.INITIALIZING, source)
            self._set_state(LifecycleState.INITIALIZED, source)
            self._notify_listeners(LifecycleEvent.INITIALIZED, source)
    
    def start(self, source: str = None):
        """Start the component."""
        source = source or self.name
        
        with self._lock:
            if self.state != LifecycleState.INITIALIZED:
                raise RuntimeError(f"Cannot start in state: {self.state.value}")
            
            self._notify_listeners(LifecycleEvent.STARTING, source)
            self._set_state(LifecycleState.STARTING, source)
            self._set_state(LifecycleState.RUNNING, source)
            self._notify_listeners(LifecycleEvent.STARTED, source)
    
    def stop(self, source: str = None):
        """Stop the component."""
        source = source or self.name
        
        with self._lock:
            if self.state not in [LifecycleState.RUNNING, LifecycleState.STARTING]:
                raise RuntimeError(f"Cannot stop in state: {self.state.value}")
            
            self._notify_listeners(LifecycleEvent.STOPPING, source)
            self._set_state(LifecycleState.STOPPING, source)
            self._set_state(LifecycleState.STOPPED, source)
            self._notify_listeners(LifecycleEvent.STOPPED, source)
    
    def pause(self, source: str = None):
        """Pause the component."""
        source = source or self.name
        
        with self._lock:
            if self.state != LifecycleState.RUNNING:
                raise RuntimeError(f"Cannot pause in state: {self.state.value}")
            
            self._notify_listeners(LifecycleEvent.PAUSING, source)
            self._set_state(LifecycleState.PAUSED, source)
            self._notify_listeners(LifecycleEvent.PAUSED, source)
    
    def resume(self, source: str = None):
        """Resume the component."""
        source = source or self.name
        
        with self._lock:
            if self.state != LifecycleState.PAUSED:
                raise RuntimeError(f"Cannot resume in state: {self.state.value}")
            
            self._notify_listeners(LifecycleEvent.RESUMING, source)
            self._set_state(LifecycleState.RUNNING, source)
            self._notify_listeners(LifecycleEvent.RESUMED, source)
    
    def terminate(self, source: str = None):
        """Terminate the component."""
        source = source or self.name
        
        with self._lock:
            self._notify_listeners(LifecycleEvent.TERMINATING, source)
            self._set_state(LifecycleState.TERMINATED, source)
            self._notify_listeners(LifecycleEvent.TERMINATED, source)
    
    def handle_error(self, error: Exception, source: str = None):
        """Handle an error."""
        source = source or self.name
        
        with self._lock:
            self._notify_listeners(LifecycleEvent.ERROR, source, error=error)
            
            if self.state not in [LifecycleState.STOPPED, LifecycleState.TERMINATED]:
                self._set_state(LifecycleState.ERROR, source)
    
    def get_state(self) -> LifecycleState:
        """Get the current state."""
        return self.state
    
    def get_event_history(self, limit: int = 100) -> List[LifecycleEventInfo]:
        """Get the event history."""
        with self._lock:
            return self.event_history[-limit:]


class ManagedAgent(Agent, LifecycleListener):
    """Agent with lifecycle management."""
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lifecycle_manager = LifecycleManager(name or f"agent_{self.name}")
        self.lifecycle_manager.add_listener(self)
        self._initialized = False
    
    def initialize(self):
        """Initialize the agent."""
        if not self._initialized:
            self.lifecycle_manager.initialize()
            self._initialized = True
    
    def start(self):
        """Start the agent."""
        if not self._initialized:
            self.initialize()
        
        self.lifecycle_manager.start()
    
    def stop(self):
        """Stop the agent."""
        self.lifecycle_manager.stop()
    
    def pause(self):
        """Pause the agent."""
        self.lifecycle_manager.pause()
    
    def resume(self):
        """Resume the agent."""
        self.lifecycle_manager.resume()
    
    def terminate(self):
        """Terminate the agent."""
        self.lifecycle_manager.terminate()
        self._initialized = False
    
    def run(self, ledger):
        """Run the agent with lifecycle management."""
        try:
            if not self._initialized:
                self.initialize()
            
            if self.lifecycle_manager.get_state() != LifecycleState.RUNNING:
                self.start()
            
            # Execute the agent
            result = super().run(ledger)
            return result
            
        except Exception as e:
            self.lifecycle_manager.handle_error(e, self.name)
            raise
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Handle lifecycle events."""
        # Custom lifecycle event handling can be implemented here
        pass
    
    def get_lifecycle_state(self) -> LifecycleState:
        """Get the current lifecycle state."""
        return self.lifecycle_manager.get_state()


class ManagedAsyncAgent(AsyncAgent, LifecycleListener):
    """Async agent with lifecycle management."""
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lifecycle_manager = LifecycleManager(name or f"async_agent_{self.name}")
        self.lifecycle_manager.add_listener(self)
        self._initialized = False
    
    def initialize(self):
        """Initialize the agent."""
        if not self._initialized:
            self.lifecycle_manager.initialize()
            self._initialized = True
    
    def start(self):
        """Start the agent."""
        if not self._initialized:
            self.initialize()
        
        self.lifecycle_manager.start()
    
    def stop(self):
        """Stop the agent."""
        self.lifecycle_manager.stop()
    
    def pause(self):
        """Pause the agent."""
        self.lifecycle_manager.pause()
    
    def resume(self):
        """Resume the agent."""
        self.lifecycle_manager.resume()
    
    def terminate(self):
        """Terminate the agent."""
        self.lifecycle_manager.terminate()
        self._initialized = False
    
    async def run(self, ledger):
        """Run the agent with lifecycle management."""
        try:
            if not self._initialized:
                self.initialize()
            
            if self.lifecycle_manager.get_state() != LifecycleState.RUNNING:
                self.start()
            
            # Execute the agent
            result = await super().run(ledger)
            return result
            
        except Exception as e:
            self.lifecycle_manager.handle_error(e, self.name)
            raise
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Handle lifecycle events."""
        # Custom lifecycle event handling can be implemented here
        pass
    
    def get_lifecycle_state(self) -> LifecycleState:
        """Get the current lifecycle state."""
        return self.lifecycle_manager.get_state()


class ManagedAgentFlow(AgentFlow, LifecycleListener):
    """Agent flow with lifecycle management."""
    
    def __init__(self, start=None, **kwargs):
        super().__init__(start=start, **kwargs)
        self.lifecycle_manager = LifecycleManager(name="managed_flow")
        self.lifecycle_manager.add_listener(self)
        self._initialized = False
    
    def initialize(self):
        """Initialize the flow."""
        if not self._initialized:
            self.lifecycle_manager.initialize()
            
            # Initialize all agents in the flow
            if self.start:
                self._initialize_agents(self.start)
            
            self._initialized = True
    
    def _initialize_agents(self, agent):
        """Initialize an agent and all its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            agent.initialize()
        
        for next_agent in agent._next:
            self._initialize_agents(next_agent)
    
    def start(self):
        """Start the flow."""
        if not self._initialized:
            self.initialize()
        
        self.lifecycle_manager.start()
    
    def stop(self):
        """Stop the flow."""
        self.lifecycle_manager.stop()
        
        # Stop all agents in the flow
        if self.start:
            self._stop_agents(self.start)
    
    def _stop_agents(self, agent):
        """Stop an agent and all its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            agent.stop()
        
        for next_agent in agent._next:
            self._stop_agents(next_agent)
    
    def pause(self):
        """Pause the flow."""
        self.lifecycle_manager.pause()
        
        # Pause all agents in the flow
        if self.start:
            self._pause_agents(self.start)
    
    def _pause_agents(self, agent):
        """Pause an agent and all its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            agent.pause()
        
        for next_agent in agent._next:
            self._pause_agents(next_agent)
    
    def resume(self):
        """Resume the flow."""
        self.lifecycle_manager.resume()
        
        # Resume all agents in the flow
        if self.start:
            self._resume_agents(self.start)
    
    def _resume_agents(self, agent):
        """Resume an agent and all its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            agent.resume()
        
        for next_agent in agent._next:
            self._resume_agents(next_agent)
    
    def terminate(self):
        """Terminate the flow."""
        self.lifecycle_manager.terminate()
        
        # Terminate all agents in the flow
        if self.start:
            self._terminate_agents(self.start)
        
        self._initialized = False
    
    def _terminate_agents(self, agent):
        """Terminate an agent and all its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            agent.terminate()
        
        for next_agent in agent._next:
            self._terminate_agents(next_agent)
    
    def run(self, ledger):
        """Run the flow with lifecycle management."""
        try:
            if not self._initialized:
                self.initialize()
            
            if self.lifecycle_manager.get_state() != LifecycleState.RUNNING:
                self.start()
            
            # Execute the flow
            result = super().run(ledger)
            return result
            
        except Exception as e:
            self.lifecycle_manager.handle_error(e, "managed_flow")
            raise
    
    async def run_async(self, ledger):
        """Run the flow asynchronously with lifecycle management."""
        try:
            if not self._initialized:
                self.initialize()
            
            if self.lifecycle_manager.get_state() != LifecycleState.RUNNING:
                self.start()
            
            # Execute the flow asynchronously
            if self.start and isinstance(self.start, ManagedAsyncAgent):
                result = await self.start.run(ledger)
            else:
                # Fallback to synchronous execution
                result = self.run(ledger)
            
            return result
            
        except Exception as e:
            self.lifecycle_manager.handle_error(e, "managed_flow")
            raise
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Handle lifecycle events."""
        # Custom lifecycle event handling can be implemented here
        pass
    
    def get_lifecycle_state(self) -> LifecycleState:
        """Get the current lifecycle state."""
        return self.lifecycle_manager.get_state()
    
    def get_agent_states(self) -> Dict[str, LifecycleState]:
        """Get the lifecycle states of all agents in the flow."""
        states = {}
        
        if self.start:
            self._collect_agent_states(self.start, states)
        
        return states
    
    def _collect_agent_states(self, agent, states: Dict[str, LifecycleState]):
        """Collect lifecycle states from an agent and its next agents."""
        if isinstance(agent, (ManagedAgent, ManagedAsyncAgent)):
            states[agent.name] = agent.get_lifecycle_state()
        
        for next_agent in agent._next:
            self._collect_agent_states(next_agent, states)


class LoggingLifecycleListener(LifecycleListener):
    """Lifecycle listener that logs events."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_default_logger()
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Log lifecycle events."""
        if event_info.error:
            self.logger.error(
                f"Lifecycle event {event_info.event.value} for {event_info.source}: {event_info.error}"
            )
        else:
            self.logger.info(
                f"Lifecycle event {event_info.event.value} for {event_info.source}"
            )


class MetricsLifecycleListener(LifecycleListener):
    """Lifecycle listener that records metrics."""
    
    def __init__(self, metrics=None):
        self.metrics = metrics or get_default_metrics()
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Record metrics for lifecycle events."""
        self.metrics.record_custom_metric(
            f"lifecycle_event_{event_info.event.value}",
            1,
            labels={"source": event_info.source}
        )
        
        # Record state transitions
        self.metrics.record_custom_metric(
            "lifecycle_state",
            1,
            labels={
                "source": event_info.source,
                "state": event_info.event.value
            }
        )


# Context manager for managed lifecycle
class ManagedLifecycle:
    """Context manager for managing lifecycle of components."""
    
    def __init__(self, components: List[Union[ManagedAgent, ManagedAsyncAgent, ManagedAgentFlow]]):
        self.components = components
    
    def __enter__(self):
        """Initialize and start all components."""
        for component in self.components:
            component.initialize()
            component.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop and terminate all components."""
        for component in self.components:
            try:
                component.stop()
            except Exception as e:
                print(f"Error stopping component {component.name}: {e}")
        
        for component in self.components:
            try:
                component.terminate()
            except Exception as e:
                print(f"Error terminating component {component.name}: {e}")
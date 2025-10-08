"""
Example demonstrating FireAnt's agent lifecycle management.
This example shows how to use lifecycle management for agents and flows.
"""

import time
import threading
from fireant import (
    Agent, AgentFlow, ManagedAgent, ManagedAsyncAgent, ManagedAgentFlow,
    LifecycleEvent, LifecycleState, LifecycleListener, LifecycleEventInfo,
    LoggingLifecycleListener, MetricsLifecycleListener, ManagedLifecycle
)


class LifecycleDataProcessor(ManagedAgent):
    """Managed agent with lifecycle awareness."""
    
    def __init__(self, name=None):
        super().__init__(name=name)
        self.processed_count = 0
    
    def execute(self, inputs):
        # Check if agent is in running state
        if self.get_lifecycle_state() != LifecycleState.RUNNING:
            print(f"⚠️ {self.name}: Not in running state, skipping execution")
            return {}
        
        # Simulate processing
        time.sleep(0.1)
        self.processed_count += 1
        
        print(f"🔄 {self.name}: Processing data (#{self.processed_count})")
        
        return {
            "processed_by": self.name,
            "count": self.processed_count,
            "state": self.get_lifecycle_state().value
        }
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Handle lifecycle events."""
        if event_info.event == LifecycleEvent.STARTED:
            print(f"🚀 {self.name}: Started processing")
        elif event_info.event == LifecycleEvent.STOPPED:
            print(f"🛑 {self.name}: Stopped processing")
        elif event_info.event == LifecycleEvent.PAUSED:
            print(f"⏸️ {self.name}: Paused processing")
        elif event_info.event == LifecycleEvent.RESUMED:
            print(f"▶️ {self.name}: Resumed processing")


class CustomLifecycleListener(LifecycleListener):
    """Custom lifecycle listener for demonstration."""
    
    def __init__(self):
        self.events = []
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Record lifecycle events."""
        self.events.append(event_info)
        print(f"📝 Custom Listener: {event_info.event.value} for {event_info.source}")


class AsyncLifecycleAgent(ManagedAsyncAgent):
    """Async managed agent with lifecycle awareness."""
    
    def __init__(self, name=None):
        super().__init__(name=name)
        self.processed_count = 0
    
    async def execute(self, inputs):
        # Check if agent is in running state
        if self.get_lifecycle_state() != LifecycleState.RUNNING:
            print(f"⚠️ {self.name}: Not in running state, skipping execution")
            return {}
        
        # Simulate async processing
        import asyncio
        await asyncio.sleep(0.1)
        self.processed_count += 1
        
        print(f"🔄 {self.name}: Async processing (#{self.processed_count})")
        
        return {
            "processed_by": self.name,
            "count": self.processed_count,
            "state": self.get_lifecycle_state().value
        }
    
    def on_lifecycle_event(self, event_info: LifecycleEventInfo):
        """Handle lifecycle events."""
        if event_info.event == LifecycleEvent.STARTED:
            print(f"🚀 {self.name}: Started async processing")
        elif event_info.event == LifecycleEvent.STOPPED:
            print(f"🛑 {self.name}: Stopped async processing")


def demonstrate_basic_lifecycle():
    """Demonstrate basic agent lifecycle management."""
    print("\n=== Basic Lifecycle Demo ===")
    
    # Create managed agent
    agent = LifecycleDataProcessor("BasicAgent")
    
    # Add custom listener
    custom_listener = CustomLifecycleListener()
    agent.lifecycle_manager.add_listener(custom_listener)
    
    # Show initial state
    print(f"📋 Initial state: {agent.get_lifecycle_state().value}")
    
    # Initialize agent
    agent.initialize()
    print(f"📋 After initialize: {agent.get_lifecycle_state().value}")
    
    # Run agent (will auto-start)
    result = agent.run({"data": "test"})
    print(f"📋 Result: {result}")
    
    # Pause agent
    agent.pause()
    print(f"📋 After pause: {agent.get_lifecycle_state().value}")
    
    # Try to run while paused (should skip)
    result = agent.run({"data": "test2"})
    print(f"📋 Result while paused: {result}")
    
    # Resume agent
    agent.resume()
    print(f"📋 After resume: {agent.get_lifecycle_state().value}")
    
    # Run again
    result = agent.run({"data": "test3"})
    print(f"📋 Result after resume: {result}")
    
    # Stop agent
    agent.stop()
    print(f"📋 After stop: {agent.get_lifecycle_state().value}")
    
    # Terminate agent
    agent.terminate()
    print(f"📋 After terminate: {agent.get_lifecycle_state().value}")
    
    # Show event history
    print(f"\n📊 Event History ({len(custom_listener.events)} events):")
    for event in custom_listener.events:
        print(f"   {event.event.value} for {event.source}")


def demonstrate_flow_lifecycle():
    """Demonstrate flow lifecycle management."""
    print("\n=== Flow Lifecycle Demo ===")
    
    # Create managed agents
    agent1 = LifecycleDataProcessor("FlowAgent1")
    agent2 = LifecycleDataProcessor("FlowAgent2")
    agent3 = LifecycleDataProcessor("FlowAgent3")
    
    # Create managed flow
    flow = ManagedAgentFlow(start=agent1.next(agent2).next(agent3))
    
    # Add logging listener
    logging_listener = LoggingLifecycleListener()
    flow.lifecycle_manager.add_listener(logging_listener)
    
    # Show initial state
    print(f"📋 Initial flow state: {flow.get_lifecycle_state().value}")
    
    # Initialize flow
    flow.initialize()
    print(f"📋 After initialize: {flow.get_lifecycle_state().value}")
    
    # Show agent states
    agent_states = flow.get_agent_states()
    print(f"📋 Agent states after initialize:")
    for name, state in agent_states.items():
        print(f"   {name}: {state.value}")
    
    # Run flow
    result = flow.run({"data": "flow_test"})
    print(f"📋 Flow result: {result}")
    
    # Pause flow
    flow.pause()
    print(f"📋 Flow paused: {flow.get_lifecycle_state().value}")
    
    # Show agent states after pause
    agent_states = flow.get_agent_states()
    print(f"📋 Agent states after pause:")
    for name, state in agent_states.items():
        print(f"   {name}: {state.value}")
    
    # Resume flow
    flow.resume()
    print(f"📋 Flow resumed: {flow.get_lifecycle_state().value}")
    
    # Run flow again
    result = flow.run({"data": "flow_test2"})
    print(f"📋 Flow result after resume: {result}")
    
    # Stop flow
    flow.stop()
    print(f"📋 Flow stopped: {flow.get_lifecycle_state().value}")
    
    # Terminate flow
    flow.terminate()
    print(f"📋 Flow terminated: {flow.get_lifecycle_state().value}")


async def demonstrate_async_lifecycle():
    """Demonstrate async agent lifecycle management."""
    print("\n=== Async Lifecycle Demo ===")
    
    # Create async managed agent
    agent = AsyncLifecycleAgent("AsyncAgent")
    
    # Add metrics listener
    metrics_listener = MetricsLifecycleListener()
    agent.lifecycle_manager.add_listener(metrics_listener)
    
    # Show initial state
    print(f"📋 Initial state: {agent.get_lifecycle_state().value}")
    
    # Initialize agent
    agent.initialize()
    print(f"📋 After initialize: {agent.get_lifecycle_state().value}")
    
    # Run agent asynchronously
    result = await agent.run({"data": "async_test"})
    print(f"📋 Async result: {result}")
    
    # Pause and resume
    agent.pause()
    print(f"📋 Paused: {agent.get_lifecycle_state().value}")
    
    agent.resume()
    print(f"📋 Resumed: {agent.get_lifecycle_state().value}")
    
    # Run again
    result = await agent.run({"data": "async_test2"})
    print(f"📋 Second async result: {result}")
    
    # Stop and terminate
    agent.stop()
    agent.terminate()
    print(f"📋 Final state: {agent.get_lifecycle_state().value}")


def demonstrate_managed_lifecycle_context():
    """Demonstrate managed lifecycle context manager."""
    print("\n=== Managed Lifecycle Context Demo ===")
    
    # Create managed agents
    agent1 = LifecycleDataProcessor("ContextAgent1")
    agent2 = LifecycleDataProcessor("ContextAgent2")
    
    # Use context manager to manage lifecycle
    with ManagedLifecycle([agent1, agent2]):
        print("🔄 Agents are now initialized and running")
        
        # Run agents
        result1 = agent1.run({"data": "context_test1"})
        result2 = agent2.run({"data": "context_test2"})
        
        print(f"📋 Results: {result1}, {result2}")
        
        # Agents will be automatically stopped and terminated when exiting context
    
    print("📋 Agents have been stopped and terminated")


def demonstrate_error_handling():
    """Demonstrate error handling in lifecycle management."""
    print("\n=== Error Handling Demo ===")
    
    class FailingAgent(ManagedAgent):
        """Agent that fails during execution."""
        
        def execute(self, inputs):
            if self.processed_count >= 2:
                raise RuntimeError("Simulated failure")
            
            self.processed_count += 1
            return {"processed": self.processed_count}
    
    # Create failing agent
    agent = FailingAgent("FailingAgent")
    
    # Add listener to capture error events
    error_listener = CustomLifecycleListener()
    agent.lifecycle_manager.add_listener(error_listener)
    
    # Run agent multiple times
    print("🔄 Running agent until failure...")
    
    for i in range(4):
        try:
            result = agent.run({"attempt": i})
            print(f"   Attempt {i+1}: Success - {result}")
        except Exception as e:
            print(f"   Attempt {i+1}: Failed - {e}")
    
    # Check final state
    print(f"📋 Final state: {agent.get_lifecycle_state().value}")
    
    # Check error events
    error_events = [e for e in error_listener.events if e.event == LifecycleEvent.ERROR]
    print(f"📋 Error events captured: {len(error_events)}")


def demonstrate_concurrent_lifecycle():
    """Demonstrate concurrent lifecycle management."""
    print("\n=== Concurrent Lifecycle Demo ===")
    
    def run_agent_with_lifecycle(agent, thread_id):
        """Run an agent in a separate thread."""
        print(f"🚀 Thread {thread_id}: Starting agent {agent.name}")
        
        # Initialize and start agent
        agent.initialize()
        agent.start()
        
        # Run agent multiple times
        for i in range(3):
            try:
                result = agent.run({"thread": thread_id, "iteration": i})
                print(f"   Thread {thread_id}, Iteration {i+1}: {result}")
                time.sleep(0.05)  # Small delay
            except Exception as e:
                print(f"   Thread {thread_id}, Iteration {i+1}: Error - {e}")
        
        # Stop and terminate agent
        agent.stop()
        agent.terminate()
        
        print(f"🏁 Thread {thread_id}: Finished agent {agent.name}")
    
    # Create multiple agents
    agents = [
        LifecycleDataProcessor(f"ConcurrentAgent{i}")
        for i in range(3)
    ]
    
    # Run agents in separate threads
    threads = []
    for i, agent in enumerate(agents):
        thread = threading.Thread(
            target=run_agent_with_lifecycle,
            args=(agent, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("📋 All concurrent executions completed")


if __name__ == "__main__":
    print("🔥 FireAnt Agent Lifecycle Examples")
    print("=" * 50)
    
    # Run all demonstrations
    demonstrate_basic_lifecycle()
    demonstrate_flow_lifecycle()
    
    # Run async demo
    import asyncio
    asyncio.run(demonstrate_async_lifecycle())
    
    demonstrate_managed_lifecycle_context()
    demonstrate_error_handling()
    demonstrate_concurrent_lifecycle()
    
    print("\n✨ Lifecycle demo completed!")
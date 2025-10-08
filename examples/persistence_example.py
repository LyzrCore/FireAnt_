"""
Example demonstrating FireAnt's state persistence capabilities.
This example shows how to save and restore agent and flow states for resilience.
"""

from fireant import (
    Agent, AgentFlow, FileStateStorage, StateManager, JSONSerializer,
    RetryPolicy, get_default_state_manager, set_default_state_manager
)
import time
import random
import os


class DataProcessor(Agent):
    """Processes data with custom state for persistence."""
    
    def __init__(self, failure_rate=0.3):
        super().__init__(
            retry_policy=RetryPolicy(max_attempts=2, delay=0.1),
            enable_persistence=True
        )
        self.failure_rate = failure_rate
        self.processed_count = 0
    
    def execute(self, inputs):
        data = inputs.get("data", [])
        
        # Simulate random failure
        if random.random() < self.failure_rate:
            raise ValueError("Random processing failure")
        
        # Simulate processing work
        time.sleep(0.2)
        
        # Update custom state
        self.processed_count += len(data)
        self.update_state("total_processed", self.processed_count)
        self.update_state("last_processed", time.time())
        
        processed = [x * 2 for x in data]
        return {
            "processed_data": processed,
            "processed_by": self.name,
            "batch_size": len(data)
        }


class DataAggregator(Agent):
    """Aggregates data with persistent state."""
    
    def __init__(self):
        super().__init__(enable_persistence=True)
        self.total_sum = 0
        self.total_count = 0
    
    def execute(self, inputs):
        processed_data = inputs.get("processed_data", [])
        
        # Simulate aggregation work
        time.sleep(0.1)
        
        # Update running totals
        batch_sum = sum(processed_data)
        batch_count = len(processed_data)
        
        self.total_sum += batch_sum
        self.total_count += batch_count
        
        # Save to persistent state
        self.update_state("running_sum", self.total_sum)
        self.update_state("running_count", self.total_count)
        self.update_state("batches_processed", self.get_state().get("batches_processed", 0) + 1)
        
        result = {
            "batch_sum": batch_sum,
            "batch_count": batch_count,
            "running_sum": self.total_sum,
            "running_count": self.total_count,
            "running_average": self.total_sum / self.total_count if self.total_count > 0 else 0
        }
        
        return {"aggregation_result": result}


class ResultReporter(Agent):
    """Reports final results."""
    
    def execute(self, inputs):
        result = inputs.get("aggregation_result", {})
        
        print(f"\n📊 Final Report:")
        print(f"   Batch sum: {result.get('batch_sum', 0)}")
        print(f"   Batch count: {result.get('batch_count', 0)}")
        print(f"   Running sum: {result.get('running_sum', 0)}")
        print(f"   Running count: {result.get('running_count', 0)}")
        print(f"   Running average: {result.get('running_average', 0):.2f}")
        
        return {"reported": True}


def demonstrate_basic_persistence():
    """Demonstrate basic state persistence."""
    print("\n=== Basic Persistence Demo ===")
    
    # Set up file-based storage
    storage_dir = "fireant_persistence_demo"
    if os.path.exists(storage_dir):
        # Clean up previous demo files
        import shutil
        shutil.rmtree(storage_dir)
    
    file_storage = FileStateStorage(storage_dir=storage_dir)
    state_manager = StateManager(file_storage)
    
    # Create a flow with persistence enabled
    flow = AgentFlow(
        start=DataProcessor(failure_rate=0.2)
            .next(DataAggregator())
            .next(ResultReporter()),
        enable_persistence=True,
        state_manager=state_manager
    )
    
    # Run the flow multiple times
    for i in range(3):
        print(f"\n🔄 Running flow iteration {i+1}")
        try:
            ledger = {"data": [1, 2, 3, 4, 5]}
            flow.run(ledger)
        except Exception as e:
            print(f"❌ Flow failed: {e}")
    
    # Check saved states
    print(f"\n💾 Saved states:")
    states = file_storage.list_states()
    for state_key in states:
        print(f"   {state_key}")
    
    return storage_dir, state_manager


def demonstrate_state_recovery():
    """Demonstrate state recovery after failure."""
    print("\n=== State Recovery Demo ===")
    
    # Set up storage
    storage_dir = "fireant_recovery_demo"
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)
    
    file_storage = FileStateStorage(storage_dir=storage_dir)
    state_manager = StateManager(file_storage)
    
    # Create agents with persistence
    processor = DataProcessor(failure_rate=0.5)  # High failure rate
    aggregator = DataAggregator()
    
    # Create flow
    flow = AgentFlow(
        start=processor.next(aggregator),
        enable_persistence=True,
        state_manager=state_manager
    )
    
    # Try to run multiple times, some will fail
    execution_id = None
    for i in range(5):
        print(f"\n🔄 Attempt {i+1}")
        try:
            ledger = {"data": [10, 20, 30]}
            flow.run(ledger)
            
            # Get the execution ID from the ledger
            execution_id = ledger.get("_flow_execution_id")
            print(f"✅ Success! Execution ID: {execution_id}")
            break
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            if not execution_id:
                execution_id = f"recovery_{int(time.time())}"
    
    # Demonstrate loading saved state
    if execution_id:
        print(f"\n🔍 Loading saved state for execution: {execution_id}")
        
        # Load processor state
        processor_state = state_manager.load_agent_state("DataProcessor", execution_id)
        if processor_state:
            print(f"   Processor state: {processor_state.custom_state}")
        
        # Load aggregator state
        aggregator_state = state_manager.load_agent_state("DataAggregator", execution_id)
        if aggregator_state:
            print(f"   Aggregator state: {aggregator_state.custom_state}")
        
        # Load flow state
        flow_state = state_manager.load_flow_state(execution_id)
        if flow_state:
            print(f"   Flow state: {flow_state.metadata}")
    
    return storage_dir, state_manager


def demonstrate_resume_execution():
    """Demonstrate resuming execution from saved state."""
    print("\n=== Resume Execution Demo ===")
    
    # Set up storage
    storage_dir = "fireant_resume_demo"
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)
    
    file_storage = FileStateStorage(storage_dir=storage_dir)
    state_manager = StateManager(file_storage)
    
    # Create a long-running flow
    class SlowProcessor(Agent):
        def __init__(self, steps=3):
            super().__init__(enable_persistence=True)
            self.steps = steps
        
        def execute(self, inputs):
            current_step = self.get_state().get("current_step", 0)
            
            if current_step >= self.steps:
                return {"completed": True, "steps_completed": current_step}
            
            # Simulate work
            time.sleep(0.5)
            current_step += 1
            
            # Update state
            self.update_state("current_step", current_step)
            self.update_state("last_step_time", time.time())
            
            print(f"   Step {current_step}/{self.steps} completed")
            
            if current_step < self.steps:
                # Simulate failure to trigger resume
                if current_step == 2:
                    raise ValueError("Simulated failure for resume demo")
            
            return {"step_completed": current_step, "steps_remaining": self.steps - current_step}
    
    # Create flow
    processor = SlowProcessor(steps=3)
    flow = AgentFlow(
        start=processor,
        enable_persistence=True,
        state_manager=state_manager
    )
    
    # First execution attempt (will fail)
    print("\n🚀 First execution attempt:")
    try:
        ledger = {}
        execution_id = flow.run(ledger).get("_flow_execution_id")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        # Get execution ID from the flow
        execution_id = f"resume_{int(time.time())}"
        # Manually save state for demo
        flow.save_flow_state(ledger, execution_id)
    
    # Resume from saved state
    print(f"\n🔄 Resuming execution from saved state:")
    try:
        # Create a new flow instance
        resumed_flow = AgentFlow(
            start=SlowProcessor(steps=3),
            enable_persistence=True,
            state_manager=state_manager
        )
        
        # Resume from state
        ledger = resumed_flow.resume_from_state(execution_id)
        print(f"✅ Resumed successfully!")
        
    except Exception as e:
        print(f"❌ Resume failed: {e}")
    
    return storage_dir, state_manager


def demonstrate_custom_storage():
    """Demonstrate using custom storage backends."""
    print("\n=== Custom Storage Demo ===")
    
    from fireant import MemoryStateStorage, PickleSerializer
    
    # Create memory-based storage
    memory_storage = MemoryStateStorage()
    state_manager = StateManager(memory_storage)
    
    # Create flow with memory storage
    flow = AgentFlow(
        start=DataProcessor(failure_rate=0.1)
            .next(DataAggregator()),
        enable_persistence=True,
        state_manager=state_manager
    )
    
    # Run flow
    print("🚀 Running flow with memory storage:")
    try:
        ledger = {"data": [5, 10, 15]}
        flow.run(ledger)
        print("✅ Flow completed with memory storage")
    except Exception as e:
        print(f"❌ Flow failed: {e}")
    
    # List states in memory
    states = memory_storage.list_states()
    print(f"\n💾 States in memory: {len(states)}")
    for state_key in states:
        print(f"   {state_key}")
    
    # Demonstrate Pickle serializer
    print(f"\n📦 Demonstrating Pickle serializer:")
    pickle_storage = FileStateStorage(
        storage_dir="fireant_pickle_demo",
        serializer=PickleSerializer()
    )
    pickle_manager = StateManager(pickle_storage)
    
    # Create flow with pickle storage
    pickle_flow = AgentFlow(
        start=DataProcessor(failure_rate=0.1),
        enable_persistence=True,
        state_manager=pickle_manager
    )
    
    try:
        ledger = {"data": [100, 200, 300]}
        pickle_flow.run(ledger)
        print("✅ Flow completed with pickle storage")
    except Exception as e:
        print(f"❌ Flow failed: {e}")


if __name__ == "__main__":
    print("🔥 FireAnt State Persistence Examples")
    print("=" * 50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run demonstrations
    storage_dir1, manager1 = demonstrate_basic_persistence()
    storage_dir2, manager2 = demonstrate_state_recovery()
    storage_dir3, manager3 = demonstrate_resume_execution()
    demonstrate_custom_storage()
    
    print("\n✨ Persistence demo completed!")
    print("📁 Check the created directories for saved state files:")
    print(f"   - {storage_dir1}")
    print(f"   - {storage_dir2}")
    print(f"   - {storage_dir3}")
    print("   - fireant_pickle_demo")
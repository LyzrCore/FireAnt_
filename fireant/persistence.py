"""
FireAnt agent state persistence capabilities.
Provides mechanisms to save and restore agent states for resilience and recovery.
"""

import json
import pickle
import os
import time
from typing import Any, Dict, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
import threading


class StateSerializer(Protocol):
    """Protocol for state serialization strategies."""
    
    def serialize(self, state: Dict[str, Any]) -> bytes:
        """Serialize state to bytes."""
        ...
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to state."""
        ...


class JSONSerializer:
    """JSON-based state serializer."""
    
    def serialize(self, state: Dict[str, Any]) -> bytes:
        return json.dumps(state, default=str).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode('utf-8'))


class PickleSerializer:
    """Pickle-based state serializer for complex objects."""
    
    def serialize(self, state: Dict[str, Any]) -> bytes:
        return pickle.dumps(state)
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        return pickle.loads(data)


@dataclass
class AgentState:
    """Represents the state of an agent."""
    agent_name: str
    agent_class: str
    status: str
    ledger_state: Dict[str, Any]
    custom_state: Dict[str, Any]
    timestamp: float
    execution_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class FlowState:
    """Represents the state of an entire flow."""
    flow_id: str
    execution_id: str
    agent_states: List[AgentState]
    current_agent_index: int
    flow_status: str
    timestamp: float
    metadata: Dict[str, Any]


class StateStorage(ABC):
    """Abstract base class for state storage backends."""
    
    @abstractmethod
    def save_state(self, key: str, state: Union[AgentState, FlowState]) -> bool:
        """Save state with given key."""
        pass
    
    @abstractmethod
    def load_state(self, key: str) -> Optional[Union[AgentState, FlowState]]:
        """Load state by key."""
        pass
    
    @abstractmethod
    def delete_state(self, key: str) -> bool:
        """Delete state by key."""
        pass
    
    @abstractmethod
    def list_states(self, prefix: str = "") -> List[str]:
        """List all state keys with given prefix."""
        pass


class FileStateStorage(StateStorage):
    """File-based state storage."""
    
    def __init__(self, storage_dir: str = "fireant_states", 
                 serializer: StateSerializer = JSONSerializer()):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.serializer = serializer
        self._lock = threading.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a state key."""
        # Sanitize key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_key}.state"
    
    def save_state(self, key: str, state: Union[AgentState, FlowState]) -> bool:
        """Save state to file."""
        try:
            with self._lock:
                file_path = self._get_file_path(key)
                
                # Convert to dict for serialization
                if isinstance(state, AgentState):
                    state_dict = asdict(state)
                else:
                    state_dict = asdict(state)
                
                # Serialize and write to file
                serialized_data = self.serializer.serialize(state_dict)
                file_path.write_bytes(serialized_data)
                
                return True
        except Exception as e:
            print(f"Error saving state {key}: {e}")
            return False
    
    def load_state(self, key: str) -> Optional[Union[AgentState, FlowState]]:
        """Load state from file."""
        try:
            with self._lock:
                file_path = self._get_file_path(key)
                
                if not file_path.exists():
                    return None
                
                # Read and deserialize
                serialized_data = file_path.read_bytes()
                state_dict = self.serializer.deserialize(serialized_data)
                
                # Determine type and reconstruct
                if "agent_states" in state_dict:
                    # It's a FlowState
                    agent_states = [
                        AgentState(**agent_state_dict) 
                        for agent_state_dict in state_dict["agent_states"]
                    ]
                    state_dict["agent_states"] = agent_states
                    return FlowState(**state_dict)
                else:
                    # It's an AgentState
                    return AgentState(**state_dict)
                    
        except Exception as e:
            print(f"Error loading state {key}: {e}")
            return None
    
    def delete_state(self, key: str) -> bool:
        """Delete state file."""
        try:
            with self._lock:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
                return True
        except Exception as e:
            print(f"Error deleting state {key}: {e}")
            return False
    
    def list_states(self, prefix: str = "") -> List[str]:
        """List all state files."""
        try:
            with self._lock:
                states = []
                for file_path in self.storage_dir.glob("*.state"):
                    key = file_path.stem.replace("_", "/")
                    if prefix and not key.startswith(prefix):
                        continue
                    states.append(key)
                return sorted(states)
        except Exception as e:
            print(f"Error listing states: {e}")
            return []


class MemoryStateStorage(StateStorage):
    """In-memory state storage for testing and temporary use."""
    
    def __init__(self):
        self._states: Dict[str, Union[AgentState, FlowState]] = {}
        self._lock = threading.Lock()
    
    def save_state(self, key: str, state: Union[AgentState, FlowState]) -> bool:
        """Save state to memory."""
        try:
            with self._lock:
                self._states[key] = state
                return True
        except Exception as e:
            print(f"Error saving state {key}: {e}")
            return False
    
    def load_state(self, key: str) -> Optional[Union[AgentState, FlowState]]:
        """Load state from memory."""
        try:
            with self._lock:
                return self._states.get(key)
        except Exception as e:
            print(f"Error loading state {key}: {e}")
            return None
    
    def delete_state(self, key: str) -> bool:
        """Delete state from memory."""
        try:
            with self._lock:
                if key in self._states:
                    del self._states[key]
                return True
        except Exception as e:
            print(f"Error deleting state {key}: {e}")
            return False
    
    def list_states(self, prefix: str = "") -> List[str]:
        """List all state keys."""
        try:
            with self._lock:
                states = list(self._states.keys())
                if prefix:
                    states = [s for s in states if s.startswith(prefix)]
                return sorted(states)
        except Exception as e:
            print(f"Error listing states: {e}")
            return []


class StateManager:
    """Manages agent and flow state persistence."""
    
    def __init__(self, storage: StateStorage, auto_save: bool = True):
        self.storage = storage
        self.auto_save = auto_save
        self._lock = threading.Lock()
    
    def save_agent_state(self, agent: 'Agent', ledger: Dict[str, Any], 
                        execution_id: Optional[str] = None) -> bool:
        """Save agent state."""
        try:
            with self._lock:
                # Extract agent-specific state
                custom_state = {}
                if hasattr(agent, 'get_state'):
                    custom_state = agent.get_state()
                
                # Create agent state
                agent_state = AgentState(
                    agent_name=agent.name,
                    agent_class=agent.__class__.__name__,
                    status=agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                    ledger_state=ledger.copy(),
                    custom_state=custom_state,
                    timestamp=time.time(),
                    execution_id=execution_id,
                    retry_count=len([h for h in getattr(agent, 'execution_history', []) 
                                   if h.get('status') == 'retry']),
                    error_message=None
                )
                
                # Generate key
                key = f"agent_{agent.name}_{execution_id or int(time.time())}"
                
                return self.storage.save_state(key, agent_state)
        except Exception as e:
            print(f"Error saving agent state: {e}")
            return False
    
    def load_agent_state(self, agent_name: str, execution_id: str) -> Optional[AgentState]:
        """Load agent state."""
        try:
            key = f"agent_{agent_name}_{execution_id}"
            state = self.storage.load_state(key)
            
            if isinstance(state, AgentState):
                return state
            return None
        except Exception as e:
            print(f"Error loading agent state: {e}")
            return None
    
    def save_flow_state(self, flow: 'AgentFlow', ledger: Dict[str, Any], 
                       execution_id: str, current_agent_index: int = 0) -> bool:
        """Save flow state."""
        try:
            with self._lock:
                # Collect states of all agents in the flow
                agent_states = []
                
                if flow.start:
                    # Traverse the flow and collect agent states
                    visited = set()
                    queue = [(flow.start, 0)]
                    
                    while queue:
                        agent, index = queue.pop(0)
                        if agent not in visited:
                            visited.add(agent)
                            
                            # Extract agent state
                            custom_state = {}
                            if hasattr(agent, 'get_state'):
                                custom_state = agent.get_state()
                            
                            agent_state = AgentState(
                                agent_name=agent.name,
                                agent_class=agent.__class__.__name__,
                                status=agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                                ledger_state={},  # Will be populated from main ledger
                                custom_state=custom_state,
                                timestamp=time.time(),
                                execution_id=execution_id,
                                retry_count=len([h for h in getattr(agent, 'execution_history', []) 
                                               if h.get('status') == 'retry'])
                            )
                            agent_states.append(agent_state)
                            
                            # Add next agents to queue
                            for next_agent in agent._next:
                                queue.append((next_agent, index + 1))
                
                # Create flow state
                flow_state = FlowState(
                    flow_id=f"flow_{execution_id}",
                    execution_id=execution_id,
                    agent_states=agent_states,
                    current_agent_index=current_agent_index,
                    flow_status="running",
                    timestamp=time.time(),
                    metadata={
                        "total_agents": len(agent_states),
                        "ledger_size": len(str(ledger))
                    }
                )
                
                # Save flow state
                key = f"flow_{execution_id}"
                return self.storage.save_state(key, flow_state)
        except Exception as e:
            print(f"Error saving flow state: {e}")
            return False
    
    def load_flow_state(self, execution_id: str) -> Optional[FlowState]:
        """Load flow state."""
        try:
            key = f"flow_{execution_id}"
            state = self.storage.load_state(key)
            
            if isinstance(state, FlowState):
                return state
            return None
        except Exception as e:
            print(f"Error loading flow state: {e}")
            return None
    
    def delete_state(self, key: str) -> bool:
        """Delete any state by key."""
        return self.storage.delete_state(key)
    
    def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """Clean up old states."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            deleted_count = 0
            
            # List all states
            all_states = self.storage.list_states()
            
            for state_key in all_states:
                state = self.storage.load_state(state_key)
                if state and hasattr(state, 'timestamp') and state.timestamp < cutoff_time:
                    if self.storage.delete_state(state_key):
                        deleted_count += 1
            
            return deleted_count
        except Exception as e:
            print(f"Error cleaning up old states: {e}")
            return 0


# Global state manager instance
_default_state_manager: Optional[StateManager] = None


def get_default_state_manager() -> StateManager:
    """Get the default state manager."""
    global _default_state_manager
    if _default_state_manager is None:
        storage = FileStateStorage()
        _default_state_manager = StateManager(storage)
    return _default_state_manager


def set_default_state_manager(state_manager: StateManager):
    """Set the default state manager."""
    global _default_state_manager
    _default_state_manager = state_manager
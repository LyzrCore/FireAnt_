from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, TypeAlias
import heapq
import logging

Ledger: TypeAlias = Dict[str, Any]

class Agent:
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._next: List[Agent] = []
        self._event_bus: Optional[EventBus] = None

    def execute(self, inputs: Ledger) -> Ledger:
        raise NotImplementedError(f"{self.name} must implement execute().")

    def prepare(self, ledger: Ledger) -> Ledger:
        return ledger.copy()

    def next(self, *agents: Agent) -> Agent:
        self._next.extend(agents)
        return self

    def run(self, ledger: Ledger) -> Ledger:
        try:
            inputs = self.prepare(ledger)
            outputs = self.execute(inputs) or {}
            ledger.update(outputs)
        except Exception as e:
            logging.error(f"Error in agent {self.name}: {e}", exc_info=True)
            return ledger

        for agent in self._next:
            agent.run(ledger)
        return ledger

    @property
    def event_bus(self) -> EventBus:
        if not self._event_bus:
            self._event_bus = EventBus()
        return self._event_bus

    def __repr__(self):
        next_names = [a.name for a in self._next]
        return f"<Agent {self.name} -> {next_names}>"


class AgentFlow:
    def __init__(self, start: Optional[Agent] = None):
        self.start = start
        self.agents: set[Agent] = set()

    def run(self, ledger: Ledger) -> Ledger:
        if self.start:
            return self.start.run(ledger)
        return ledger

    def add_parallel_branch(self, agents: List[Agent]) -> AgentFlow:
        self.agents.update(agents)
        return self

    def add_conditional_branch(self, name: str, agent: Agent) -> AgentFlow:
        setattr(self, f"cond_{name}", agent)
        return self

    def add_checkpoint(self, name: str, cond: Callable[[Ledger], bool]) -> AgentFlow:
        setattr(self, f"chk_{name}", cond)
        return self

    def __repr__(self):
        return f"<AgentFlow start={self.start} agents={len(self.agents)}>"


class ManagerAgent(Agent):
    def __init__(self, worker_pool: Optional[WorkerPool] = None, **kwargs):
        super().__init__(**kwargs)
        self.worker_pool = worker_pool or WorkerPool()
        self.tasks: List[tuple[int, str, Ledger]] = []  # (priority, worker_name, data)

    def assign_task(self, name: str, data: Ledger, priority: int = 1):
        heapq.heappush(self.tasks, (-priority, name, data))  # negative for max-heap

    def add_worker(self, name: str, worker: Agent):
        self.worker_pool[name] = worker

    def process_tasks(self, ledger: Ledger):
        while self.tasks:
            _, name, data = heapq.heappop(self.tasks)
            worker = self.worker_pool.get(name)
            if worker:
                try:
                    worker.run({**ledger, **data})
                except Exception as e:
                    logging.error(f"Worker {name} failed: {e}", exc_info=True)


class WorkerPool(Dict[str, Agent]):
    pass


class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Agent]] = {}

    def subscribe(self, event: str, agent: Agent):
        self.subscribers.setdefault(event, []).append(agent)

    def unsubscribe(self, event: str, agent: Agent):
        if event in self.subscribers:
            self.subscribers[event] = [a for a in self.subscribers[event] if a != agent]

    def publish(self, event: str, data: Ledger):
        for agent in self.subscribers.get(event, []):
            try:
                agent.run(data)
            except Exception as e:
                logging.error(f"Error in subscriber {agent.name} for event {event}: {e}", exc_info=True)


class EventAgent(Agent):
    def publish(self, event: str, data: Ledger):
        if self.event_bus:
            self.event_bus.publish(event, data)

    # def execute(self, inputs: Ledger) -> Ledger:
    #     return {}

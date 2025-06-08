from typing import Any, Callable, Dict, List

class Agent:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self._next = []
        self._event_bus = None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def prepare(self, ledger: Dict[str, Any]) -> Dict[str, Any]:
        return ledger

    def next(self, *agents):
        self._next.extend(agents)
        return self

    def run(self, ledger: Dict[str, Any]):
        inputs = self.prepare(ledger)
        outputs = self.execute(inputs)
        ledger.update(outputs or {})
        for agent in self._next:
            agent.run(ledger)
        return ledger

    @property
    def event_bus(self):
        if not self._event_bus:
            self._event_bus = EventBus()
        return self._event_bus
class AgentFlow:
    def __init__(self, start=None):
        self.start = start
        self.agents = set()

    def run(self, ledger: Dict[str, Any]):
        if self.start:
            return self.start.run(ledger)
        return ledger

    def add_parallel_branch(self, agents: List[Agent]):
        self.agents.update(agents)
        return self

    def add_conditional_branch(self, name: str, agent: Agent):
        setattr(self, f"cond_{name}", agent)
        return self

    def add_checkpoint(self, name: str, cond: Callable[[Dict[str, Any]], bool]):
        setattr(self, f"chk_{name}", cond)
        return self
class ManagerAgent(Agent):
    def __init__(self, worker_pool=None, **kwargs):
        super().__init__(**kwargs)
        self.worker_pool = worker_pool or {}
        self.tasks = []

    def assign_task(self, name, data, priority=1):
        self.tasks.append((priority, name, data))

    def add_worker(self, name, worker):
        self.worker_pool[name] = worker

    def process_tasks(self, ledger):
        for _, name, data in sorted(self.tasks, reverse=True):
            if name in self.worker_pool:
                self.worker_pool[name].run({**ledger, **data})
class WorkerPool(dict): 
    pass
class EventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event, agent):
        self.subscribers.setdefault(event, []).append(agent)

    def publish(self, event, data):
        for agent in self.subscribers.get(event, []):
            agent.run(data)
            
class EventAgent(Agent):
    def publish(self, event, data):
        if self.event_bus:
            self.event_bus.publish(event, data)
# FireAnt 🐜🔥

**A minimal, powerful agent orchestration framework in just 85 lines of code.**

Like fire ants combining to float on water or defend against threats, FireAnt enables simple agents to work together and accomplish complex tasks through emergent collaboration.

<img width="1880" height="902" alt="CleanShot 2025-10-05 at 12 44 13@2x" src="https://github.com/user-attachments/assets/e3ebf906-00fa-4c4c-b399-2fd6db4081a2" />

---

## 🔥 Why FireAnt?

FireAnt is designed for the **agentic coding era**—where LLM-powered platforms like Claude Code and GPT Codex build applications by orchestrating specialized agents rather than generating monolithic code.

### The Agentic Advantage

**Traditional Approach:**

LLM → Generate complete app → Hope it works → Debug black box

**FireAnt Approach:**

LLM → Compose specialized agents → Trace execution flow → Improve incrementally

With FireAnt, you get:
- ✅ **Traceability**: See exactly which agent did what
- ✅ **Modularity**: Replace or improve individual agents without breaking the system
- ✅ **Debuggability**: Inspect the ledger at any point in the workflow
- ✅ **Composability**: Combine simple agents to create complex behaviors
- ✅ **Lightweight**: No heavy dependencies, just pure Python patterns

---

## 🚀 Quick Start
```python
from fireant import Agent, AgentFlow

class DataFetcher(Agent):
    def execute(self, inputs):
        return {"data": [1, 2, 3, 4, 5]}

class DataProcessor(Agent):
    def execute(self, inputs):
        data = inputs.get("data", [])
        return {"result": sum(data)}

class ResultLogger(Agent):
    def execute(self, inputs):
        print(f"Final result: {inputs['result']}")
        return {}

# Chain agents together
flow = AgentFlow(
    start=DataFetcher()
        .next(DataProcessor())
        .next(ResultLogger())
)

# Run the workflow
flow.run(ledger={})
# Output: Final result: 15
```

## 🏗️ Core Concepts
The Ledger Pattern
FireAnt uses a shared ledger (dictionary) that flows through your agent pipeline. Each agent reads from it, processes data, and writes back to it—accumulating context as it goes.

```python
ledger = {}
agent1.run(ledger)  # ledger now has agent1's outputs
agent2.run(ledger)  # agent2 sees agent1's results + adds its own
```

### Agent Chaining
Build sequential pipelines by chaining agents:

```python
workflow = (
    FetchAgent()
    .next(ValidateAgent())
    .next(TransformAgent())
    .next(SaveAgent())
)
```

### Task Delegation
Use ManagerAgent to delegate work to a pool of specialized workers:

```python
manager = ManagerAgent()
manager.add_worker("data_processing", DataWorker())
manager.add_worker("validation", ValidationWorker())

manager.assign_task("data_processing", {"file": "data.csv"}, priority=2)
manager.assign_task("validation", {"schema": "user"}, priority=1)

manager.process_tasks(ledger)  # Executes by priority
```

### Event-Driven Communication
Decouple agents using the event bus:

```python
class TriggerAgent(EventAgent):
    def execute(self, inputs):
        self.publish("data_ready", {"dataset": "users"})
        return {}

class ListenerAgent(Agent):
    def execute(self, inputs):
        print(f"Received: {inputs}")
        return {}

listener = ListenerAgent()
trigger = TriggerAgent()
trigger.event_bus.subscribe("data_ready", listener)

trigger.run({})  # ListenerAgent automatically runs when event fires
```

## 📦 Installation
```python
# Clone the repository
git clone https://github.com/yourusername/fireant.git
cd fireant

# FireAnt has zero dependencies - just copy fireant.py to your project!
cp fireant.py your_project/```

Or install via pip (coming soon):
```python
pip install fireant
```
## 💡 Real-World Example: Building a Web Scraper
```python
class URLFetcher(Agent):
    def execute(self, inputs):
        url = inputs.get("url")
        # Fetch content logic here
        return {"html": "<html>...</html>"}

class HTMLParser(Agent):
    def execute(self, inputs):
        html = inputs.get("html", "")
        # Parse HTML logic here
        return {"data": {"title": "Example", "content": "..."}}

class DataValidator(Agent):
    def execute(self, inputs):
        data = inputs.get("data", {})
        is_valid = "title" in data and "content" in data
        return {"valid": is_valid}

class ErrorHandler(Agent):
    def execute(self, inputs):
        if not inputs.get("valid"):
            print("Error: Invalid data structure!")
        return {}

class DataSaver(Agent):
    def execute(self, inputs):
        if inputs.get("valid"):
            # Save to database
            print(f"Saved: {inputs['data']['title']}")
        return {}

# Build the workflow
scraper = AgentFlow(
    start=URLFetcher()
        .next(HTMLParser())
        .next(DataValidator())
        .next(ErrorHandler(), DataSaver())  # Parallel execution
)

scraper.run({"url": "https://example.com"})
```
## 🎯 Use Cases

LLM Agent Orchestration: Build complex AI agent systems where each agent has a specific role
Data Pipelines: Create ETL workflows with clear separation of concerns
Workflow Automation: Automate business processes with traceable, debuggable flows
Microservice Coordination: Orchestrate service calls in a clean, maintainable way
Event-Driven Systems: Build reactive applications with loose coupling

## 🧬 Philosophy
FireAnt embraces the Unix philosophy for agent systems:

Each agent does one thing well
Agents work together seamlessly
The ledger is a universal interface

Just like fire ants achieve remarkable feats through simple individual behaviors and effective collaboration, FireAnt agents combine to solve complex problems through clear composition patterns.

## 🤝 Contributing
Contributions are welcome! FireAnt is intentionally minimal, but improvements to the core patterns, bug fixes, and documentation are always appreciated.
```bash
# Fork the repo, create a branch, make your changes
git checkout -b feature/amazing-feature
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
# Open a Pull Request
```
Built for the age of LLM-powered development. 85 lines. Infinite possibilities.

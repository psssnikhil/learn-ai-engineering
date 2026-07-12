---
title: Agent Communication & Coordination
description: >-
  Master how agents communicate, share information, and coordinate actions in
  multi-agent systems
duration: 40 min
difficulty: intermediate
has_code: false
module: module-12
youtube: 'https://www.youtube.com/watch?v=L4-Z0K77bgk'
---
# Agent Communication & Coordination

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Agent communication patterns | 40 min | Intermediate |
| Message passing vs shared memory | | |
| Coordination strategies | | |
| Implement agent communication | | |

---

## 📚 Why Agent Communication Matters

In multi-agent systems, **how agents communicate determines system effectiveness**. Poor communication leads to:
- ❌ Duplicated work
- ❌ Conflicting actions
- ❌ Missed information
- ❌ System failures

Good communication enables:
- ✅ Coordinated actions
- ✅ Information sharing
- ✅ Efficient task completion
- ✅ Robust systems

---

## 🔄 Communication Patterns

### 1. Direct Messaging (Point-to-Point)

```
Agent A ────message────> Agent B
         <────reply─────
```

**How it works:**
- Agent A sends message directly to Agent B
- Agent B processes and responds
- Clear, explicit communication

**When to use:**
- Few agents (2-5)
- Clear agent relationships
- Simple workflows

**Example:**
```python
class ResearchAgent:
    def request_info(self, query):
        message = {
            "from": "ResearchAgent",
            "to": "DataAgent",
            "type": "query",
            "content": query
        }
        return self.send(message)

class DataAgent:
    def handle_message(self, message):
        if message["type"] == "query":
            data = self.search_database(message["content"])
            return {
                "from": "DataAgent",
                "to": message["from"],
                "type": "response",
                "data": data
            }
```

### 2. Broadcast (One-to-Many)

```
         Agent A (broadcasts)
              |
     ┌────────┼────────┐
     ↓        ↓        ↓
  Agent B  Agent C  Agent D
```

**How it works:**
- One agent sends message to all others
- All agents receive and decide if relevant
- Good for announcements

**When to use:**
- System-wide updates
- Event notifications
- Status changes

**Example:**
```python
class OrchestratorAgent:
    def broadcast_task(self, task):
        message = {
            "from": "Orchestrator",
            "type": "new_task",
            "task": task,
            "priority": "high"
        }
        # Send to all agents
        for agent in self.agent_pool:
            agent.receive(message)
```

### 3. Publish-Subscribe

```
              ┌─────────────┐
              │   Message   │
              │    Bus      │
              └──────┬──────┘
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
  Subscriber A  Subscriber B  Subscriber C
  (interested)  (interested)  (not listening)
```

**How it works:**
- Agents publish messages to topics
- Other agents subscribe to topics of interest
- Decoupled communication

**When to use:**
- Large numbers of agents
- Dynamic agent pools
- Event-driven architectures

**Example:**
```python
class MessageBus:
    def __init__(self):
        self.subscribers = {}  # topic -> [agents]
    
    def subscribe(self, topic, agent):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(agent)
    
    def publish(self, topic, message):
        if topic in self.subscribers:
            for agent in self.subscribers[topic]:
                agent.receive(message)

# Usage
bus = MessageBus()
bus.subscribe("data_ready", analysis_agent)
bus.subscribe("data_ready", visualization_agent)

data_agent.publish(bus, "data_ready", processed_data)
```

### 4. Request-Reply Pattern

```
Agent A ──── request ───→ Agent B
                          ↓ (process)
Agent A ←─── reply ─────  Agent B
```

**How it works:**
- Synchronous communication
- Requestor waits for response
- Clear transaction boundaries

**Example:**
```python
class ValidationAgent:
    async def validate_output(self, output):
        request = {
            "type": "validate",
            "data": output,
            "rules": self.validation_rules
        }
        # Wait for response
        response = await self.send_and_wait(request)
        return response["is_valid"]
```

---

## 🗂️ Shared Memory vs Message Passing

### Message Passing

```
Agent A                    Agent B
   │                          │
   │ ─────message───────────> │
   │                          │
   │ <────response───────────  │
```

**Advantages:**
- ✅ Explicit communication
- ✅ Easy to debug
- ✅ Clear information flow
- ✅ Works across distributed systems

**Disadvantages:**
- ❌ Can be slower
- ❌ Message overhead
- ❌ Serialization costs

### Shared Memory

```
┌─────────────────────────┐
│    Shared Memory/DB     │
│  ┌──────────────────┐   │
│  │ State, Data, etc │   │
│  └──────────────────┘   │
└───────┬─────────┬───────┘
        ↓         ↓
   Agent A    Agent B
   (reads/    (reads/
    writes)    writes)
```

**Advantages:**
- ✅ Fast access
- ✅ No message overhead
- ✅ Easy data sharing
- ✅ Persistent state

**Disadvantages:**
- ❌ Race conditions possible
- ❌ Harder to debug
- ❌ Synchronization needed
- ❌ Coupling between agents

---

## 📊 Communication Pattern Comparison

| Pattern | Best For | Latency | Complexity | Scalability |
|---------|----------|---------|------------|-------------|
| **Direct Messaging** | Small teams | Low | Low | Poor |
| **Broadcast** | Announcements | Medium | Low | Medium |
| **Pub-Sub** | Event-driven | Medium | Medium | **Excellent** |
| **Request-Reply** | Transactions | Low-Med | Low | Medium |
| **Shared Memory** | Fast access | **Lowest** | High | Poor |

---

## 🎯 Coordination Strategies

### 1. Centralized Coordination (Orchestrator)

```python
class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "analysis": AnalysisAgent(),
            "report": ReportAgent()
        }
        self.task_queue = []
    
    async def execute_workflow(self, task):
        # Orchestrator decides the workflow
        
        # Step 1: Research
        research_data = await self.agents["research"].execute(task)
        
        # Step 2: Analysis  
        analysis = await self.agents["analysis"].execute(research_data)
        
        # Step 3: Report
        report = await self.agents["report"].execute(analysis)
        
        return report
```

**Advantages:**
- Clear control flow
- Easy to understand
- Centralized decision-making

**Disadvantages:**
- Single point of failure
- Orchestrator can become bottleneck
- Less flexible

### 2. Decentralized Coordination (Peer-to-Peer)

```python
class CollaborativeAgent:
    def __init__(self, role):
        self.role = role
        self.peers = []
    
    async def execute(self, task):
        # Decide what to do based on task and role
        if self.can_handle(task):
            result = await self.process(task)
            return result
        else:
            # Find appropriate peer
            peer = self.find_best_peer(task)
            return await peer.execute(task)
    
    def can_handle(self, task):
        return task.type in self.capabilities
```

**Advantages:**
- No single point of failure
- Flexible and adaptive
- Scales well

**Disadvantages:**
- Complex to implement
- Harder to debug
- Possible inefficiencies

### 3. Contract Net Protocol

```
Orchestrator: "Who can handle task X?"
    ↓
Agent A: "I can, cost=10, time=5min"
Agent B: "I can, cost=8, time=7min"
Agent C: "Can't handle this"
    ↓
Orchestrator: "Agent B, you're selected!"
    ↓
Agent B: Executes task
    ↓
Agent B: Returns result
```

**Implementation:**
```python
class TaskManager:
    async def assign_task(self, task):
        # 1. Announce task
        bids = []
        for agent in self.agent_pool:
            bid = await agent.bid_on_task(task)
            if bid:
                bids.append((agent, bid))
        
        # 2. Select best bid
        if bids:
            best_agent, best_bid = min(bids, key=lambda x: x[1]["cost"])
            
            # 3. Award contract
            result = await best_agent.execute_task(task)
            return result
        
        return None

class WorkerAgent:
    async def bid_on_task(self, task):
        if not self.can_handle(task):
            return None
        
        return {
            "cost": self.estimate_cost(task),
            "time": self.estimate_time(task),
            "quality": self.expected_quality(task)
        }
```

---

## 💻 Implementation Example: Complete System

```python
from typing import Dict, List, Any
from dataclasses import dataclass
import asyncio

@dataclass
class Message:
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: float

class MessageBus:
    """Central communication hub"""
    def __init__(self):
        self.subscribers: Dict[str, List] = {}
        self.message_history: List[Message] = []
    
    def subscribe(self, topic: str, agent):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(agent)
    
    async def publish(self, topic: str, message: Message):
        self.message_history.append(message)
        
        if topic in self.subscribers:
            tasks = [
                agent.receive_message(message) 
                for agent in self.subscribers[topic]
            ]
            await asyncio.gather(*tasks)

class BaseAgent:
    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.message_bus = message_bus
        self.inbox: List[Message] = []
    
    async def send_message(self, recipient: str, msg_type: str, content: Any):
        message = Message(
            sender=self.name,
            recipient=recipient,
            message_type=msg_type,
            content=content,
            timestamp=time.time()
        )
        await self.message_bus.publish(recipient, message)
    
    async def receive_message(self, message: Message):
        self.inbox.append(message)
        await self.process_message(message)
    
    async def process_message(self, message: Message):
        raise NotImplementedError

class ResearchAgent(BaseAgent):
    async def process_message(self, message: Message):
        if message.message_type == "research_request":
            query = message.content
            results = await self.conduct_research(query)
            
            await self.send_message(
                recipient=message.sender,
                msg_type="research_results",
                content=results
            )
    
    async def conduct_research(self, query: str):
        # Simulate research
        await asyncio.sleep(1)
        return f"Research results for: {query}"

class AnalysisAgent(BaseAgent):
    async def process_message(self, message: Message):
        if message.message_type == "analysis_request":
            data = message.content
            analysis = await self.analyze(data)
            
            await self.send_message(
                recipient=message.sender,
                msg_type="analysis_results",
                content=analysis
            )
    
    async def analyze(self, data: str):
        # Simulate analysis
        await asyncio.sleep(1)
        return f"Analysis of: {data}"

class OrchestratorAgent(BaseAgent):
    async def process_message(self, message: Message):
        if message.message_type == "task_request":
            await self.execute_workflow(message.content)
    
    async def execute_workflow(self, task: str):
        print(f"📋 Starting workflow for: {task}")
        
        # Step 1: Request research
        await self.send_message("research_agent", "research_request", task)
        
        # Wait for research results
        research_msg = await self.wait_for_message("research_results")
        print(f"📚 Received research: {research_msg.content}")
        
        # Step 2: Request analysis
        await self.send_message("analysis_agent", "analysis_request", research_msg.content)
        
        # Wait for analysis results
        analysis_msg = await self.wait_for_message("analysis_results")
        print(f"🔍 Received analysis: {analysis_msg.content}")
        
        print("✅ Workflow complete!")
    
    async def wait_for_message(self, msg_type: str, timeout: float = 10.0):
        start = time.time()
        while time.time() - start < timeout:
            for msg in self.inbox:
                if msg.message_type == msg_type:
                    self.inbox.remove(msg)
                    return msg
            await asyncio.sleep(0.1)
        raise TimeoutError(f"No message of type {msg_type} received")

# Usage Example
async def main():
    bus = MessageBus()
    
    # Create agents
    orchestrator = OrchestratorAgent("orchestrator", bus)
    research = ResearchAgent("research_agent", bus)
    analysis = AnalysisAgent("analysis_agent", bus)
    
    # Subscribe to topics
    bus.subscribe("orchestrator", orchestrator)
    bus.subscribe("research_agent", research)
    bus.subscribe("analysis_agent", analysis)
    
    # Start workflow
    await orchestrator.execute_workflow("AI trends in 2024")

# Run
asyncio.run(main())
```

---

## 🎓 Key Takeaways

```
✅ Choose communication pattern based on system needs
✅ Direct messaging: simple, few agents
✅ Pub-Sub: scalable, many agents, events
✅ Shared memory: fast but needs synchronization
✅ Coordination strategy depends on control requirements
✅ Orchestrator: centralized, predictable
✅ Decentralized: flexible, fault-tolerant
```

---

## 📊 Pattern Selection Guide

| Your Need | Best Pattern | Example |
|-----------|--------------|---------|
| **2-3 agents, simple** | Direct messaging | Chatbot + Memory |
| **5-10 agents** | Pub-Sub | Multi-tool agent |
| **10+ agents** | **Pub-Sub + Orchestrator** | Enterprise system |
| **Real-time coordination** | Shared memory | Trading system |
| **Fault tolerance needed** | Decentralized | Distributed AI |
| **Clear workflow** | Orchestrator | Data pipeline |

---

## 💡 Best Practices

1. **Keep messages small**: Only send necessary data
2. **Use timeouts**: Don't wait forever for responses
3. **Log communication**: Essential for debugging
4. **Handle failures**: Agents may not respond
5. **Version messages**: Allow schema evolution
6. **Validate inputs**: Don't trust all messages
7. **Monitor latency**: Track communication bottlenecks

---

## 🚀 Next Lesson

**Lesson 3: Orchestrator-Worker Patterns** - Deep dive into coordinating agent teams

You'll learn:
- 🎯 Building robust orchestrators
- 🔄 Dynamic task delegation
- 📊 Load balancing across agents
- ⚡ Optimizing workflow execution

**Get ready to build production-ready multi-agent systems!** 🏗️

---

## 📚 Additional Resources

- 📺 [Agent Communication Patterns](https://www.youtube.com/watch?v=L4-Z0K77bgk)
- 📄 [LangGraph: Agent Communication](https://langchain-ai.github.io/langgraph/)
- 💻 [CrewAI Communication Docs](https://docs.crewai.com/)
- 📖 [AutoGen Message Passing](https://microsoft.github.io/autogen/docs/tutorial/introduction)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*

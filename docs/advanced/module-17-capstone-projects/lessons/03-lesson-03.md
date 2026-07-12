---
title: 'Project 2: Autonomous Coding Agent'
description: >-
  Build an AI agent that reads code, identifies bugs, suggests fixes, runs
  tests, and creates pull requests using the ReAct pattern and tool use
duration: 150 min
difficulty: advanced
has_code: false
module: module-17
youtube: 'https://www.youtube.com/watch?v=DjuXACWYkkU'
objectives:
  - Implement the ReAct loop for iterative code analysis
  - 'Build tools for file reading, code execution, and git operations'
  - Design prompts that guide systematic debugging
  - Create a working agent that can fix simple bugs autonomously
---
# Project 2: Autonomous Coding Agent

## Project Overview

Build an AI-powered coding agent that can:
- Analyze a codebase to understand structure and dependencies
- Identify bugs and potential issues
- Suggest and apply fixes
- Run tests to verify changes
- Create git commits with descriptive messages

**Time estimate**: 15-20 hours
**Skills used**: AI Agents, Tool Use, ReAct Pattern, Code Analysis

---

## Architecture

```
User: "Fix the failing tests in src/utils.py"
    |
    v
[Agent Loop (ReAct)]
    |
    |-- THINK: "I need to understand what tests are failing"
    |-- ACT: run_tests() -> see failures
    |
    |-- THINK: "Test X fails because function Y has a bug"
    |-- ACT: read_file("src/utils.py") -> see the code
    |
    |-- THINK: "The bug is on line 42, off-by-one error"
    |-- ACT: edit_file("src/utils.py", line=42, new_code=...)
    |
    |-- THINK: "Let me verify the fix"
    |-- ACT: run_tests() -> all pass
    |
    |-- THINK: "Tests pass. I'll commit the fix."
    |-- ACT: git_commit("Fix off-by-one error in utils.py")
    v
Done: "Fixed the bug. All tests pass."
```

---

## Step 1: Define Agent Tools

```python
import subprocess
import os
import json

class CodingTools:
    """Tools available to the coding agent."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
    
    def read_file(self, filepath: str) -> str:
        """Read the contents of a file."""
        full_path = os.path.join(self.repo_path, filepath)
        if not os.path.exists(full_path):
            return f"Error: File {filepath} does not exist"
        with open(full_path, "r") as f:
            lines = f.readlines()
        # Return with line numbers
        return "".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    
    def list_files(self, directory: str = ".") -> str:
        """List files in a directory."""
        full_path = os.path.join(self.repo_path, directory)
        if not os.path.isdir(full_path):
            return f"Error: {directory} is not a directory"
        entries = []
        for item in sorted(os.listdir(full_path)):
            item_path = os.path.join(full_path, item)
            prefix = "[DIR]" if os.path.isdir(item_path) else "[FILE]"
            entries.append(f"  {prefix} {item}")
        return "
".join(entries)
    
    def edit_file(self, filepath: str, old_text: str, new_text: str) -> str:
        """Replace old_text with new_text in a file."""
        full_path = os.path.join(self.repo_path, filepath)
        if not os.path.exists(full_path):
            return f"Error: File {filepath} does not exist"
        
        with open(full_path, "r") as f:
            content = f.read()
        
        if old_text not in content:
            return f"Error: Could not find the specified text in {filepath}"
        
        content = content.replace(old_text, new_text, 1)
        
        with open(full_path, "w") as f:
            f.write(content)
        
        return f"Successfully edited {filepath}"
    
    def run_tests(self, test_path: str = "") -> str:
        """Run tests and return results."""
        cmd = ["python", "-m", "pytest", "-v"]
        if test_path:
            cmd.append(test_path)
        
        result = subprocess.run(
            cmd, cwd=self.repo_path,
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        # Truncate if too long
        if len(output) > 3000:
            output = output[:1500] + "
... (truncated) ...
" + output[-1500:]
        return output
    
    def run_command(self, command: str) -> str:
        """Run a shell command (limited to safe commands)."""
        safe_prefixes = ["python", "grep", "find", "cat", "wc", "git diff", "git log"]
        if not any(command.startswith(p) for p in safe_prefixes):
            return "Error: Command not allowed for safety"
        
        result = subprocess.run(
            command, shell=True, cwd=self.repo_path,
            capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    
    def git_commit(self, message: str) -> str:
        """Stage all changes and commit."""
        subprocess.run(["git", "add", "-A"], cwd=self.repo_path)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_path, capture_output=True, text=True
        )
        return result.stdout + result.stderr
```

---

## Step 2: Build the ReAct Agent Loop

```python
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are an expert coding agent. You analyze code, find bugs, and fix them.

You have access to these tools:
- read_file(filepath): Read a file's contents with line numbers
- list_files(directory): List files in a directory
- edit_file(filepath, old_text, new_text): Replace text in a file
- run_tests(test_path): Run tests and see results
- run_command(command): Run safe shell commands
- git_commit(message): Commit all changes

Use the ReAct pattern:
1. THINK: Reason about what to do next
2. ACT: Call a tool with specific arguments
3. OBSERVE: Read the tool result
4. Repeat until the task is complete

When you're done, respond with DONE: followed by a summary.

Always:
- Read the failing test first to understand what's expected
- Read the source code before making changes
- Make minimal, targeted fixes
- Run tests after each change to verify
- Commit with a clear, descriptive message"""

def run_agent(task: str, tools: CodingTools, max_steps: int = 15):
    """Run the coding agent on a task."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]
    
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file with line numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}},
                    "required": ["filepath"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {"directory": {"type": "string", "default": "."}},
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace old_text with new_text in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"}
                    },
                    "required": ["filepath", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run pytest tests",
                "parameters": {
                    "type": "object",
                    "properties": {"test_path": {"type": "string", "default": ""}},
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit all changes",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"]
                }
            }
        }
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tool_definitions,
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # Check if agent is done
        if msg.content and "DONE:" in msg.content:
            print(f"Agent completed in {step + 1} steps")
            print(msg.content)
            return msg.content
        
        # Execute tool calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                
                print(f"  Step {step + 1}: {fn_name}({fn_args})")
                
                # Call the appropriate tool
                tool_fn = getattr(tools, fn_name, None)
                if tool_fn:
                    result = tool_fn(**fn_args)
                else:
                    result = f"Error: Unknown tool {fn_name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
    
    return "Agent reached maximum steps without completing"
```

---

## Step 3: Usage Example

```python
# Initialize tools for a repository
tools = CodingTools(repo_path="/path/to/your/project")

# Run the agent
result = run_agent(
    task="The tests in tests/test_utils.py are failing. Read the test file and the source code, identify the bugs, fix them, verify the tests pass, and commit your changes.",
    tools=tools
)
```

---

## Evaluation

Test your agent on these scenarios:

| Scenario | Description | Success Criteria |
|----------|-------------|-----------------|
| Simple bug fix | Off-by-one error in a loop | Tests pass after fix |
| Missing import | Function uses unimported module | Tests pass, clean import |
| Logic error | Wrong conditional operator | Tests pass, minimal change |
| Type error | String/int confusion | Tests pass, type-correct |

---

## Extension Ideas

- Add a `search_code` tool for grep/ripgrep across the codebase
- Implement multi-file refactoring support
- Add GitHub API integration to create PRs
- Build a web UI for interacting with the agent
- Add memory so the agent remembers past fixes and patterns

---

## Resources

- **OpenAI Function Calling Guide**: Tool use with GPT models
- **ReAct Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **LangChain Agents**: Framework for building tool-using agents
- **Aider**: Open-source AI pair programming tool for reference

---

## Next Project

**Project 3: Multi-Agent Research System** — Build a team of AI agents that collaborate to research topics, write reports, and fact-check each other.

# HR Job Creation Flow

An AI-powered job posting assistant built with [CrewAI Flows](https://docs.crewai.com/en/concepts/flows). It collects role details conversationally, researches the market and relevant AI skills in parallel, then generates a polished, company-tailored job posting — all orchestrated through a stateful, persistent flow.

## How It Works

The flow accepts a user message and routes it to one of three paths based on intent:

```
User message
     │
     ▼
┌──────────┐
│  Router   │  ← Structured LLM output (RouterIntent)
└────┬─────┘
     │
     ├── "conversation"  →  Collect missing info (job_role, location, company_name)
     │                      and reply naturally
     │
     ├── "job_creation"  →  Kick off HrCrew (3 agents research & write in parallel)
     │                      and return the finished posting
     │
     └── "refinement"    →  A standalone editor agent applies targeted changes
                            to the existing posting
```

1. **Conversation** — If any of the three required fields (`job_role`, `location`, `company_name`) are missing, the router returns `"conversation"` and the LLM generates a friendly reply asking for the missing details.
2. **Job Creation** — Once all three fields are collected, the `HrCrew` kicks off. Two research agents (market research + AI skills) run in parallel, then a writer agent synthesizes everything into a complete job posting.
3. **Refinement** — When a posting already exists and the user gives feedback, a lightweight standalone agent edits the posting without re-running the full crew.

State is persisted across sessions with `@persist()`, so the conversation can be resumed later.

## Architecture Deep-Dive

### Flow Orchestration

The `HrJobCreationFlow` class uses CrewAI's [Flow](https://docs.crewai.com/en/concepts/flows) decorators to define the execution graph:

- **`@start()`** — `starting_flow()` adds the user message to history and kicks off routing.
- **`@router(starting_flow)`** — `routing_intent()` returns one of three string literals (`"job_creation"`, `"conversation"`, `"refinement"`) that determine the next step.
- **`@listen("conversation" | "job_creation" | "refinement")`** — Three listener methods each handle their respective path.

```python
# main.py — simplified flow skeleton
@persist()
class HrJobCreationFlow(Flow[FlowState]):
    @start()
    def starting_flow(self): ...

    @router(starting_flow)
    def routing_intent(self): ...      # returns "conversation" | "job_creation" | "refinement"

    @listen("conversation")
    def follow_up_conversation(self): ...

    @listen("job_creation")
    def handle_job_creation(self): ...

    @listen("refinement")
    def handle_refinement(self): ...
```

### Structured State with Pydantic

The flow's state is a [Pydantic `BaseModel`](https://docs.crewai.com/en/guides/flows/mastering-flow-state) that tracks everything across turns:

| Field | Type | Purpose |
|-------|------|---------|
| `user_message` | `str` | Current user input |
| `message_history` | `List[Message]` | Full conversation log with timestamps |
| `role_info` | `RoleInfo` | Collected fields: `job_role`, `location`, `company_name` |
| `job_posting` | `Optional[str]` | The generated (or refined) posting |
| `feedback` | `Optional[str]` | User's refinement feedback |
| `answer_message` | `Optional[str]` | Conversational reply from the router |

### Structured LLM Output for Routing

The router uses a `RouterIntent` Pydantic model as the LLM's `response_format`, so the output is parsed and validated automatically — no fragile string matching:

```python
class RouterIntent(BaseModel):
    user_intent: Literal["job_creation", "conversation", "refinement"]
    role_info: Optional[RoleInfo] = Field(default_factory=RoleInfo)
    feedback: Optional[str] = None
    answer_message: Optional[str] = None
    reasoning: str
```

The LLM is called with `LLM(model="gpt-5-nano", response_format=RouterIntent)`, and the response is a typed `RouterIntent` instance.

### Crew Embedded in a Flow

The `HrCrew` is a standalone `@CrewBase` crew with its own YAML configuration. The flow calls it in one line:

```python
@listen("job_creation")
def handle_job_creation(self):
    crew = HrCrew().crew()
    result = crew.kickoff(inputs={
        "job_role": self.state.role_info.job_role,
        "location": self.state.role_info.location,
        "company_name": self.state.role_info.company_name,
    })
    self.state.job_posting = result.raw
```

### Async Parallel Research

Inside `HrCrew`, the two research tasks use `async_execution=True` so they run concurrently. The writing task depends on both via `context=[...]`, creating a fan-out/fan-in pattern:

```
job_market_research_task ──┐
  (async_execution=True)   ├──→ job_posting_creation_task
ai_skills_research_task ───┘       (waits for both)
  (async_execution=True)
```

### Standalone Agent for Refinement

Refinement doesn't need a full crew — it's a single task. The flow creates an `Agent` directly and calls `agent.kickoff(prompt)`:

```python
@listen("refinement")
def handle_refinement(self):
    agent = Agent(
        role="Senior Job Posting Editor",
        goal="Refine and improve job postings based on specific feedback",
        backstory="...",
        tools=[FirecrawlSearchTool()],
        llm=self.llm,
    )
    result = agent.kickoff(prompt)
    self.state.job_posting = result.raw
```

This is intentionally lightweight — no YAML config, no crew overhead.

### YAML-First Agent & Task Config

Agents and tasks for the `HrCrew` are defined in YAML, not Python. The Python crew class just wires them together:

**`config/agents.yaml`** defines three agents:
- `job_market_researcher` — searches for real job postings and extracts market patterns
- `ai_skills_researcher` — identifies relevant AI tools and competencies for the role
- `job_posting_writer` — researches the company and synthesizes everything into a posting

**`config/tasks.yaml`** defines three tasks with `{job_role}`, `{location}`, and `{company_name}` interpolation variables.

### State Persistence

The `@persist()` decorator on the flow class saves state between sessions, so a conversation can be resumed across multiple runs.

## Project Structure

```
hr_job_creation/
├── pyproject.toml                          # Dependencies, entry points, crewai config
├── .env                                    # API keys (OPENAI_API_KEY, FIRECRAWL_API_KEY)
├── README.md
└── src/
    └── hr_job_creation/
        ├── __init__.py
        ├── main.py                         # Flow orchestration, state models, router
        ├── crews/
        │   └── hr_crew/
        │       ├── __init__.py
        │       ├── hr_crew.py              # @CrewBase crew with 3 agents, 3 tasks
        │       └── config/
        │           ├── agents.yaml         # Agent roles, goals, backstories
        │           └── tasks.yaml          # Task descriptions, expected outputs
        └── tools/
            ├── __init__.py
            └── custom_tool.py              # Custom tool template (placeholder)
```

## Getting Started

### Prerequisites

- Python >= 3.10, < 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key
- A [Firecrawl](https://www.firecrawl.dev/) API key

### Install

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
crewai install
# or
uv sync
```

### Environment Setup

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_key_here
FIRECRAWL_API_KEY=your_key_here
```

### Run

```bash
crewai run
```

The default state starts with: *"HI, create a job posting for a data engineer based in NYC for Johnson & Johnson"* — which has all three fields populated, so it routes straight to job creation.

You can also visualize the flow graph:

```bash
crewai flow plot
```

## Configuration

### Changing the LLM Model

The router and refinement agent use `gpt-5-nano` by default, set in `main.py`:

```python
class HrJobCreationFlow(Flow[FlowState]):
    llm = LLM(model="gpt-5-nano")
```

Change the model string to use a different provider or model.

### Modifying Agents & Tasks

Edit the YAML files to change agent behavior without touching Python:

- **`src/hr_job_creation/crews/hr_crew/config/agents.yaml`** — Adjust roles, goals, and backstories
- **`src/hr_job_creation/crews/hr_crew/config/tasks.yaml`** — Change task descriptions, expected outputs, or add new sections to the posting

Variables like `{job_role}`, `{location}`, and `{company_name}` are interpolated from the flow state at kickoff time.

### Adding a New Crew

1. Create a new directory under `src/hr_job_creation/crews/`
2. Add `config/agents.yaml` and `config/tasks.yaml`
3. Create a `@CrewBase` crew class
4. Call it from a flow listener in `main.py` via `YourCrew().crew().kickoff(inputs={...})`

## Example Usage

```
User:  Hi there!
       → Routes to "conversation"
Agent: Hello! I'm your HR job creation assistant. I can help you craft
       a professional job posting. To get started, I'll need:
       - The job role (e.g., "Software Engineer")
       - The location (e.g., "Remote", "NYC")
       - The company name
       What role are you hiring for?

User:  I need a Marketing Manager posting for Acme Corp
       → Routes to "conversation" (location still missing)
Agent: Great — a Marketing Manager at Acme Corp! Where will this role
       be based?

User:  London
       → Routes to "job_creation" (all 3 fields collected)
       → HrCrew kicks off:
         1. Market researcher finds real Marketing Manager postings in London
         2. AI skills researcher identifies relevant AI tools for marketing
         3. Writer researches Acme Corp and produces the full posting
Agent: [Complete job posting in markdown]

User:  Can you make the tone more casual and add remote flexibility?
       → Routes to "refinement"
Agent: [Updated posting with casual tone and remote option added]
```

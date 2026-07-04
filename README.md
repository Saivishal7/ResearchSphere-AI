# ResearchSphere AI
### Terminal-Based Agentic AI Research Matching & Collaboration Assistant

**ResearchSphere AI** is a terminal-based, multi-agent AI assistant designed for university ecosystems to bridge the gap between academic intent and high-impact research collaboration. It serves two main target audiences:
1. **Students**: Helps them discover suitable faculty supervisors aligned with their research interests and methodology preferences.
2. **Professors**: Enables them to identify emerging research trends, explore domain gaps, and locate complementary cross-department collaborators.

Built with a highly structured, multi-agent LangGraph system and backed by semantic retrieval (RAG) over ChromaDB, ResearchSphere AI ensures deep relevance, citation-enforced answers, and safety through a human-in-the-loop confirmation gate for all consequential actions.

---

## 📂 Folder Structure

The project has been initialized with the following clean, modular structure:

```
researchsphere-ai/
├── main.py                      # CLI entrypoint — the chat loop
├── config.py                    # Environment configuration loading & validation
├── .env.example                 # Environment template with API placeholders
├── requirements.txt             # Project library dependencies
├── setup.sh                     # Automatic setup script for Linux/Mac
├── setup.bat                    # Automatic setup script for Windows
│
├── graph/                       # LangGraph Orchestration
│   ├── __init__.py
│   ├── state.py                 # Conversation State Schema (TypedDict)
│   ├── build_graph.py           # LangGraph StateGraph wiring (nodes + edges)
│   └── router.py                # Router node logic
│
├── agents/                      # Specialized Agent Modules
│   ├── __init__.py
│   ├── student_agent.py         # Student interest profile building
│   ├── professor_agent.py       # Professor task orchestrator
│   ├── faculty_retrieval_agent.py # Semantic search & re-ranking over ChromaDB
│   ├── trend_agent.py           # Established and emerging trends analysis
│   ├── gap_agent.py             # Under-explored research gaps analyst
│   ├── collaboration_agent.py   # Complementary cross-department pairings
│   ├── project_recommendation_agent.py # Concrete proposal creation
│   └── confirmation_agent.py    # Human-in-the-loop approval gate
│
├── tools/                       # Integrations, Client Wrappers, and Helpers
│   ├── __init__.py
│   ├── gemini_tool.py           # Gemini SDK model calling wrapper
│   ├── embedding_tool.py        # Local Sentence-Transformers embeddings
│   ├── chroma_tool.py           # ChromaDB collection client and querying
│   ├── tavily_tool.py           # Tavily search integration
│   ├── semantic_scholar_tool.py # Semantic Scholar citation details loader
│   └── email_tool.py            # Outgoing messaging mock/stub wrapper
│
├── data/                        # Local Raw Datasets
│   └── faculty/                 # Faculty JSON profiles directory
│
├── ingestion/                   # DB Ingestion Pipelines
│   ├── __init__.py
│   └── load_chroma.py           # One-time script to chunk and index profiles
│
├── prompts/                     # Modular System Prompt Templates
│   ├── __init__.py
│   ├── router_prompt.py
│   ├── student_prompt.py
│   ├── professor_prompt.py
│   ├── trend_prompt.py
│   ├── collaboration_prompt.py
│   └── project_prompt.py
│
└── chroma_store/                # ChromaDB persistent local data store (auto-created)
```

---

## 🛠️ Folder Purpose Explanations

- **`graph/`**: Contains the core graph routing logic and states. Separating state, structure, and nodes allows for clean modifications to conversational flows without changing agent logic.
- **`agents/`**: Contains small, decoupled Python files representing individual specialized AI roles. Each agent consumes a portion of the shared State, performs its task, and returns updates.
- **`tools/`**: Encapsulates external libraries (ChromaDB, Gemini SDK, Tavily Web Search, Semantic Scholar Academic API, and outgoing communications). No business logic lives here; they are simple, robust utility wrappers.
- **`data/`**: Stores raw institutional datasets (such as faculty bio files) in structured JSON formats.
- **`ingestion/`**: Holds scripts designed for loading, cleaning, embedding, and storing dataset records into local vector storage.
- **`prompts/`**: Keeps prompt text completely separated from python logic, allowing prompt engineering and tuning to occur without modifying execution paths.

---

## 📦 Dependencies

All dependencies are defined in `requirements.txt`:
* **Core Orchestration**: `langgraph`, `langchain`, `pydantic`
* **AI & Language Models**: `google-generativeai`, `langchain-google-genai`
* **Vector Embeddings & Database**: `chromadb`, `sentence-transformers`
* **APIs & Protocols**: `tavily-python`, `requests`, `python-dotenv`
* **Terminal Interface**: `rich`

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.11+** installed on your system.

### 2. Quick-Start Shell Scripts (Recommended)
We provide automated setup scripts that create a virtual environment, install all dependencies, and print configuration guidelines.

* **On macOS / Linux**:
  ```bash
  chmod +x setup.sh
  ./setup.sh
  ```

* **On Windows**:
  ```cmd
  setup.bat
  ```

### 3. Manual Installation (Optional fallback)
If you prefer to configure the environment yourself:
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

---

## ⚙️ Configuration Setup

1. Copy `.env.example` to a new `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your API credentials:
   * **`GEMINI_API_KEY`**: Obtain from Google AI Studio.
   * **`TAVILY_API_KEY`**: Obtain from Tavily Developer Portal.
   * **`SEMANTIC_SCHOLAR_API_KEY`**: Optional, can be left blank for default public-tier usage.

---

## 🎮 How to Run

To run the initialized configuration check:
```bash
python main.py
```
This will print the configuration health check status and verify that all directories, environments, and packages are fully prepared for subsequent development phases!

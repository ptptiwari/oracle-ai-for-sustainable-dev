# Oracledb-MCP

An AI-powered Oracle database assistant that uses natural language to interact with Oracle databases through MCP (Model Context Protocol).

## Features

- **Natural Language to SQL**: Convert natural language queries to Oracle SQL
- **MCP Integration**: Uses Model Context Protocol for tool integration
- **Multiple LLM Providers**: Support for OpenAI, OCI, and local LLM providers
- **Vector Database**: Oracle Vector DB integration with semantic search capabilities
- **RAG System**: Advanced Retrieval Augmented Generation with Oracle AI Vector Search
- **LangChain Integration**: Seamless compatibility with LangChain ecosystem
- **Human-in-the-Loop**: Built-in feedback and validation workflows

## Prerequisites

- Python 3.8+
- Oracle Database access
- LM provider credentials
- SqlCL installation

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd oracle-mcp-demo
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. Configure the application by editing `config/config.json` with your settings

### Environment Variables

For vector database connection, set these environment variables:

```bash
export copilot_rag_connectionstring="user/password@host:port/service"
export copilot_rag_schema="your_schema_name"
```

## Usage

### Command Line Interface

Start the Oracle Mcp CLI:

```bash
# Using the installed script
oracle-mcp-demo

# Or directly with Python
python -m oracledb-mcp-demo
```

### Testing

Run tests with the provided scripts:

```bash
# Run all tests
python scripts/run_all.py tests all

# Run specific test types
python scripts/run_all.py tests unit              # Unit tests only
python scripts/run_all.py tests e2e               # End-to-end tests only

# Or use pytest directly
pytest tests/unit/ -v                    # Unit tests
pytest tests/e2e/ -v                     # E2E tests
```

## Architecture

Key components:

- **CLI Interface**: Rich command-line interface for user interaction
- **Task Orchestrator**: Performance-optimized task execution and workflow management
- **RAG System**: Advanced Retrieval Augmented Generation with Oracle Vector DB
- **LLM Providers**: Integration with various language models (OpenAI, OCI, Local)
- **MCP Client**: Model Context Protocol client for tool integration
- **Vector Database**: Oracle Vector Store with semantic search and embedding providers

## Development

### Project Structure

```bash
oracledb-mcp-demo/
├── application/          # CLI application
├── core/                # Core business logic
├── llm/                 # LLM provider implementations
├── mcp_client/          # MCP client implementation
├── vector_db/           # Oracle Vector Store integration
├── tools/               # Utility tools for knowledge management
├── tests/               # Comprehensive test suite
└── config/              # Configuration files
```

### Code Quality

Maintain code quality with built-in tools:

```bash
# Check and fix markdown quality
python scripts/run_all.py lint --fix

# Preview fixes before applying
python scripts/run_all.py lint --fix --dry-run
```

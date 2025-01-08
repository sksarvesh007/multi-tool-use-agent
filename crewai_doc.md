# CrewAI System with FastAPI Integration

## System Overview

This system implements an AI-powered assistant using CrewAI framework, integrating multiple tools for handling various types of queries including weather information, RAG-based document retrieval, and web searches. The system is exposed through a FastAPI interface for easy HTTP access.

## Architecture Components

### 1. Core Components

- **LLM Integration**: Uses Google's Gemini 1.5 Flash 8B model
- **Vector Store**: Pinecone for document storage and retrieval
- **Search Integration**: Tavily for web searches
- **Weather API**: OpenWeather API for weather data
- **FastAPI**: REST API interface

### 2. Tools Implementation

#### RAG (Retrieval-Augmented Generation) Tool

```python
def rag_retrieve_enhance_generate(state: str) -> str
```

- Uses Pinecone vector store for document retrieval
- Implements similarity search with scoring
- Processes retrieved documents through LangChain RAG prompt
- Best used for queries related to premium policies

#### Web Search Tool

```python
def web_search_enhance_return(state: str) -> str
```

- Integrates Tavily search API
- Processes search results through RAG chain
- Used for general knowledge queries not related to weather or policies

#### Weather Tool

```python
def getweather(city_name: str) -> str
```

- Integrates with OpenWeather API
- Returns weather information in metric units
- Takes city name as input

## System Setup

### Environment Variables Required

```
LANGSMITH_API_KEY
TAVILY_API_KEY
PINECONE_API_KEY
GEMINI_API_KEY
```

### Vector Store Configuration

- **Index Name**: agentic-rag
- **Namespace**: wondervector5000
- **Cloud Provider**: AWS
- **Region**: us-east-1

## API Endpoints

### POST /execute_crew

Executes queries through the CrewAI system.

**Request Format:**

```json
{
    "query": "string"
}
```

**Response Format:**

```json
{
    "verbose_logs": "string",
    "result": "string"
}
```

### GET /

Health check endpoint returning system status.

## CrewAI Agent Configuration

The system uses a single agent with the following configuration:

- **Role**: User Assistant
- **Goal**: Process user queries related to weather and mathematical operations
- **Tools**: Weather API, RAG Retrieval, Web Search
- **Model**: Gemini 1.5 Flash 8B
- **Temperature**: 0.7 for general queries, 0 for RAG operations

## Usage Examples

### Weather Query

```python
query = "Tell me about the weather in Mumbai"
# Returns current weather information for Mumbai
```

### Policy Query

```python
query = "What are the premium policy details?"
# Uses RAG tool to retrieve and process policy information
```

### General Knowledge Query

```python
query = "What are the latest developments in AI?"
# Uses web search tool to find and process current information
```

## Error Handling

The FastAPI implementation includes:

- Exception handling for failed queries
- HTTP 500 responses for system errors
- Proper error message propagation

## Performance Considerations

- RAG operations use a context window appropriate for the model
- Web searches are limited to top 5 results for efficiency
- Async support in FastAPI for better request handling

## Security Notes

- API keys should be properly secured using environment variables
- FastAPI endpoints should be properly secured in production
- Rate limiting should be implemented for production use

## Deployment

The system can be deployed using:

```bash
uvicorn crewai_fastapi:app --host 0.0.0.0 --port 8000
```

## Limitations and Considerations

1. Rate limits apply to external APIs (OpenWeather, Tavily)
2. Vector store capacity depends on Pinecone tier
3. Response times may vary based on query complexity
4. Model context window limitations apply

# LangGraph Workflow Documentation

This documentation explains the LangGraph workflow implemented in the provided Python files (`agents.py`, `graph.py`, and `graph_utils.py`). The workflow is designed to process user questions and generate responses using a combination of tools, including Retrieval-Augmented Generation (RAG), web search, and weather data retrieval. The workflow is built using the `langgraph` library, which allows for the creation of stateful, directed graphs to manage complex workflows.

---

## Table of Contents

1. **Overview**
2. **Key Components**
   * GraphState
   * Nodes and Tools
   * Workflow Structure
3. **Detailed Workflow**
   * Toolfinder
   * RAG (Retrieval-Augmented Generation)
   * Web Search
   * Weather Retrieval
   * Output Combiner
4. **API Integration**
5. **How to Run the Workflow**
6. **Conclusion**

---

## 1. Overview

The LangGraph workflow is a stateful, directed graph that processes user questions and generates responses using a combination of tools. The workflow is divided into several nodes, each responsible for a specific task, such as determining which tools to use, retrieving information from a vector store, performing web searches, fetching weather data, and combining the results into a final output.

The workflow is designed to be modular, allowing for easy extension or modification of individual components.

---

## 2. Key Components

### GraphState

The `GraphState` is a TypedDict that defines the structure of the state object passed between nodes in the workflow. It contains the following fields:

* `question`: The user's input question.
* `web_search_req`: A boolean indicating whether a web search is required.
* `rag_req`: A boolean indicating whether RAG (Retrieval-Augmented Generation) is required.
* `weather_req`: A boolean indicating whether weather data is required.
* `web_search_generation`: The output from the web search tool.
* `rag_generation`: The output from the RAG tool.
* `weather_generation`: The output from the weather tool.
* `final_generation`: The final combined output.

### Nodes and Tools

The workflow consists of the following nodes, each representing a specific task:

1. **Toolfinder** : Determines which tools (RAG, web search, weather) are needed based on the user's question.
2. **RAG (Retrieval-Augmented Generation)** : Retrieves relevant information from a vector store and generates a response.
3. **Web Search** : Performs a web search and generates a response based on the results.
4. **Weather Retrieval** : Fetches weather data for a specific city.
5. **Output Combiner** : Combines the outputs from the RAG, web search, and weather tools into a final response.

### Workflow Structure

The workflow is structured as a directed graph, where each node is connected to the next in a linear sequence:

* **START** → **Toolfinder** → **RAG** → **Web Search** → **Weather Retrieval** → **Output Combiner** → **END**

---

## 3. Detailed Workflow

### Toolfinder

The `toolfinder` node is the first step in the workflow. It analyzes the user's question and determines which tools are required:

* **RAG** : Used for questions related to motor insurance policies, premiums, and coverage.
* **Web Search** : Used for general queries that are not related to insurance or weather.
* **Weather Retrieval** : Used for questions asking about the weather in a specific city.

The `toolfinder` node updates the `GraphState` with the required tools and passes the state to the next node.

### RAG (Retrieval-Augmented Generation)

The `rag_retrieve_enhance_generate` node is responsible for retrieving relevant information from a vector store and generating a response. The vector store contains documents related to motor insurance policies. The node performs the following steps:

1. Enhances the user's question to make it more suitable for retrieval.
2. Retrieves relevant documents from the vector store.
3. Generates a response using a language model (LLM) and the retrieved documents.

The output is stored in the `rag_generation` field of the `GraphState`.

### Web Search

The `web_search_enhance_return` node performs a web search using the Tavily search tool. It enhances the user's question to extract the relevant part for the search and then generates a response based on the search results. The output is stored in the `web_search_generation` field of the `GraphState`.

### Weather Retrieval

The `get_weather` node fetches weather data for a specific city using the OpenWeatherMap API. It extracts the city name from the user's question, retrieves the weather data, and generates a response. The output is stored in the `weather_generation` field of the `GraphState`.

### Output Combiner

The `output_combiner` node combines the outputs from the RAG, web search, and weather tools into a final response. It uses a language model to restructure and combine the outputs in a way that matches the user's tone and intent. The final output is stored in the `final_generation` field of the `GraphState`.

---

## 4. API Integration

The workflow can be integrated into a FastAPI application to expose it as an API endpoint. The API accepts a user question as input, processes it through the workflow, and returns the final response. The API is defined in the `api.py` file, which wraps the workflow in a FastAPI endpoint.

---

## 5. How to Run the Workflow

To run the workflow, follow these steps:

1. Install the required dependencies:
   bash

   Copy

   ```
   pip install fastapi uvicorn langgraph langchain pinecone tavily-python python-dotenv requests
   ```
2. Set up the environment variables in a `.env` file:
   plaintext

   Copy

   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GOOGLE_API_KEY=your_google_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   ```
3. Run the FastAPI server:
   bash

   Copy

   ```
   python langgraph_fast.py
   ```
4. Send a POST request to the `/process-question/` endpoint with a JSON payload containing the user's question:
   bash

   Copy

   ```
   curl -X POST "http://127.0.0.1:8000/process-question/" -H "Content-Type: application/json" -d '{"question": "Tell me about premium policy"}'
   ```

---

## 6. Conclusion

The LangGraph workflow is a powerful tool for processing user questions and generating responses using a combination of tools. The modular design allows for easy extension or modification of individual components, making it suitable for a wide range of applications. By integrating the workflow into a FastAPI application, it can be easily exposed as an API endpoint for use in other systems.

---

This documentation provides a high-level overview of the workflow and its components. For more detailed information, refer to the code comments in the provided files.

from typing import List, TypedDict
from graph_utils import (
    question_router,
    retrieve_with_scores,
    rag_chain,
    rag_prompt_enhancer,
    fetch_weather,
    web_search_enhancer,
    web_search_tool, 
    output_combiner ,
    combine_outputs
)
from langchain.schema import Document

# Initializing the GraphState structure
class GraphState(TypedDict):
    question: str
    web_search_req: bool
    rag_req: bool
    weather_req: bool
    web_search_generation: str
    rag_generation: str
    weather_generation: str
    final_generation :str


def toolfinder(state: GraphState) -> GraphState:
    print("TOOL FINDER INVOKED")
    question = state["question"]
    tools = question_router.invoke({"question": question})
    web_tool_req = tools.web_search
    weather_tool_req = tools.get_weather
    rag_tool_req = tools.vectorstore
    return {
        "question": question,
        "web_search_req": web_tool_req,
        "rag_req": rag_tool_req,
        "weather_req": weather_tool_req,
    }

# agentsecond.py

import time  

def rag_retrieve_enhance_generate(state: GraphState) -> GraphState:
    question = state["question"]
    rag_tool_req = state["rag_req"]
    if rag_tool_req:
        print("RAG TOOL INVOKED")
        question_new = rag_prompt_enhancer.invoke({"question": question})
        print(question_new)
        documents = retrieve_with_scores(question_new.rag_enhanced_query)
        generation = ""
        for chunk in rag_chain.stream({"context": documents, "question": question_new}):
            generation += chunk
            print(chunk, end="", flush=True)  # Stream tokens to console
            if "streamlit_callback" in state:  # Stream words to Streamlit
                for word in chunk.split():  # Split chunk into words
                    state["streamlit_callback"](f"**RAG Tool Output:** {generation} {word}")
        print()  # New line after streaming
        return {"question": question, "rag_generation": generation}
    return {"question": question, "rag_generation": ""}

def web_search_enhance_return(state: GraphState) -> GraphState:
    question = state["question"]
    web_search_tool_req = state["web_search_req"]
    if web_search_tool_req:
        print("WEB SEARCH INVOKED")
        web_search_query = web_search_enhancer.invoke({"question": question})
        print(web_search_query)
        docs = web_search_tool.invoke({"query": web_search_query.extracted_query})
        web_results = "\n".join([d["content"] for d in docs])
        generation = ""
        for chunk in rag_chain.stream({"context": docs, "question": web_search_query}):
            generation += chunk
            print(chunk, end="", flush=True)  # Stream tokens to console
            if "streamlit_callback" in state:  # Stream words to Streamlit
                for word in chunk.split():  # Split chunk into words
                    state["streamlit_callback"](f"**Web Search Tool Output:** {generation} {word}")
                    time.sleep(0.1)  # Add a small delay between words
        print()  # New line after streaming
        return {"question": question, "web_search_generation": generation}
    return {"question": question, "web_search_generation": ""}

def get_weather(state: GraphState) -> GraphState:
    user_input = state["question"]
    weather_tool_req = state["weather_req"]
    if weather_tool_req:
        print("WEATHER TOOL INVOKED")
        weather_response = fetch_weather(user_input)
        print(weather_response)  # Stream the weather response
        if "streamlit_callback" in state:  # Stream weather response to Streamlit
            for word in weather_response.split():  # Split response into words
                state["streamlit_callback"](f"**Weather Tool Output:** {word}")
                time.sleep(0.1)  # Add a small delay between words
        return {"question": user_input, "weather_generation": weather_response}
    return {"question": user_input, "weather_generation": ""}

def output_combiner(state:GraphState) : 
    question = state["question"]
    output1 = state["rag_generation"]
    output2 = state["weather_generation"]
    output3 = state["web_search_generation"]
    #result = combine_outputs(user_input=question , output1=output1 , output2=output2 , output3=output3)
    result = output1 + " " + output2 + " " + output3
    return {"final_generation" : result , "question" : question}
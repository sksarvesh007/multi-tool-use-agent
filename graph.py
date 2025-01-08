from langgraph.graph import END, StateGraph, START
from agents import GraphState, toolfinder, output_combiner, rag_retrieve_enhance_generate, web_search_enhance_return, get_weather

# Create a workflow with StateGraph
workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("toolfinder", toolfinder)
workflow.add_node("rag", rag_retrieve_enhance_generate)
workflow.add_node("websearch", web_search_enhance_return)
workflow.add_node("weather", get_weather)
workflow.add_node("output_combiner", output_combiner)

# Build the graph by adding edges between the nodes
workflow.add_edge(START, "toolfinder")
workflow.add_edge("toolfinder", "rag")
workflow.add_edge("rag", "websearch")
workflow.add_edge("websearch", "weather")
workflow.add_edge("weather", "output_combiner")
workflow.add_edge("output_combiner", END)

# Compile the workflow
app_graph = workflow.compile()

from pprint import pprint

# Inputs for the question
inputs = {
    "question": "Tell me about premium policy"
}  

# Function to stream reasoning and output
def stream_reasoning_and_output(state):
    print("------------ STATE AT EACH STEP ------------")
    for node, result in state.items():
        print(f"At node {node}:")
        if isinstance(result, bool):
            print(f"  - {node} request status: {result}")
        elif isinstance(result, str):
            print(f"  - {node} generation output: {result}")
        else:
            print(f"  - {node} data: {result}")
    print("------------------------------------------")
    
    print("--------------------------")
    print("FINAL OUTPUT:")
    print(state.get("final_generation", "No final output generated"))

#Invoke the app and capture the final state
final_state = app_graph.invoke(inputs)

stream_reasoning_and_output(final_state)

print("--------------------------")
print("INPUT QUESTION: ")
print(final_state["question"])
print("FINAL GENERATION")
print(final_state["final_generation"])

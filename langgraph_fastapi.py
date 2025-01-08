from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import app_graph, stream_reasoning_and_output

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for the input
class QuestionInput(BaseModel):
    question: str

# Define the API endpoint
@app.post("/process-question/")
async def process_question(input: QuestionInput):
    try:
        # Prepare the inputs for the graph
        inputs = {"question": input.question}

        # Invoke the graph and capture the final state
        final_state = app_graph.invoke(inputs)

        # Stream the reasoning and output (optional, for logging purposes)
        stream_reasoning_and_output(final_state)

        # Return the final generation
        return {
            "input_question": final_state["question"],
            "final_generation": final_state.get("final_generation", "No final output generated")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_crewai import DevelopmentCrew  # Import your crew
import io
import sys
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    verbose_logs: str
    result: str

# Initialize the CrewAI crew
crew_instance = DevelopmentCrew()

@app.post("/execute_crew", response_model=QueryResponse)
async def execute_crew(request: QueryRequest):
    try:
        # Get user query from request
        user_query = request.query

        # Capture verbose output
        old_stdout = sys.stdout  # Backup the current stdout
        sys.stdout = io.StringIO()  # Redirect stdout to capture verbose logs

        # Execute the Crew and get the result
        crew = crew_instance.crew(user_input=user_query)
        result = crew.kickoff()  # Kick off the crew and get the result

        # Capture the verbose logs
        verbose_logs = sys.stdout.getvalue()

        # Restore the original stdout
        sys.stdout = old_stdout

        # Return the verbose logs and result as a response
        return QueryResponse(verbose_logs=verbose_logs, result=str(result))
    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "CrewAI FastAPI is running!"}

# Run the app directly with Uvicorn if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

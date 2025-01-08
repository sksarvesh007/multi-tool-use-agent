from dotenv import load_dotenv
from crewai import Agent, Crew, Task
from crewai.tools import tool
from crewai.llm import LLM
import logging
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
import os 
os.environ["LITELLM_VERBOSE"] = "0"
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from typing import Dict, Optional, List
import requests
from datetime import datetime, timedelta, timezone
import time
# Initialize LLM
llm = LLM(
    model="gemini/gemini-1.5-flash-8b",
    temperature=0.7
)
load_dotenv()
llms = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,
    stream = True
)
langchain_api_key = os.getenv("LANGSMITH_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = tavily_api_key
model_name = 'multilingual-e5-large'
import asyncio

if not asyncio.get_event_loop_policy().get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())
# Initialize PineconeEmbeddings with the event loop
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "agentic-rag"
namespace = "wondervector5000"

# Initialize PineconeVectorStore
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

# Define tools

openweather_api_key = "2bbdba23aab5e01ae16edae0b35bbf0a"

def retrieve_with_scores(query, k=5):
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

prompt = hub.pull("rlm/rag-prompt" , api_key = langchain_api_key)
rag_chain = prompt | llms | StrOutputParser()


@tool
def rag_retrieve_enhance_generate(state: str) -> str:
    """This function answers the query related to the premium policies"""
    print("RAG TOOL INVOKED")
    question = state
    documents = retrieve_with_scores(question)
    doc_list = []
    for i in range(len(documents)):
        doc_list.append(documents[i][0].page_content)
    generation = rag_chain.invoke({"context":doc_list , "question":question})
    return generation


web_search_tool  = TavilySearchResults(k=5)
@tool
def web_search_enhance_return(state:str) -> str:
    """Use this tool only for answering questions not related to weather and policy"""
    question = state
    print("WEB SEARCH INVOKED")
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    generation = rag_chain.invoke({"context" : web_results , "question" : question})
    return generation

@tool 
def getweather(city_name:str) -> str: 
    """ Only provide with the city name in the input to this function"""
    print("FETCH WEATHER FUNCTION INVOKED")
    print(city_name)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={openweather_api_key}&units=metric"
    response = requests.get(url)
    response.raise_for_status()  
    weather_info = response.json()
    return str(weather_info)

# Define the crew as a class
class DevelopmentCrew:
    def __init__(self):
        self.llm = llm
        self.tools = [ getweather , rag_retrieve_enhance_generate ,web_search_enhance_return]
        self.agent = Agent(
            role="User Assistant",
            goal="Process user queries related to weather and mathematical operations.",
            backstory="""You are an expert , you are supposed to use appropriate tool for answering various questions of the user , use weather tool for answering weather related query
            .Use web search tool for when you need to access internet. Use rag retrieval tool for when the query is related to premiums and policies""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            crew_sharing=True,
        )
    
    def crew(self , user_input):
        # Task to process natural language queries
        task = Task(
            description=str(("Understand the user's query and perform the required operations." , user_input)),
            agent=self.agent,
            expected_output="A detailed response based on the user's query.",
        )
        return Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True,
        )

# Main execution
if __name__ == "__main__":
    # Natural language input
    queryone = "Tell me about the weather in Mumbai"
    # Dynamically create the crew and execute it with the input query
    result = DevelopmentCrew().crew(user_input=queryone).kickoff()
    print("\n\n---------- RESULT ----------\n\n")
    print(result)

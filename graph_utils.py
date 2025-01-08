
import asyncio
import nest_asyncio  # Import nest_asyncio to handle event loops in threads

# Apply nest_asyncio to allow event loops in threads
nest_asyncio.apply()

# Your existing imports and code...
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

# Load environment variables
load_dotenv()

# Ensure an event loop exists
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "no current event loop" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        raise e

# Your existing code...
openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
langchain_api_key = os.getenv("LANGSMITH_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = tavily_api_key
model_name = 'multilingual-e5-large'

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

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,
    stream = True
)

def retrieve_with_scores(query, k=5):
    results = vector_store.similarity_search_with_score(query, k=k)
    return results
#TOOL FINDING DONE
class RouterModel(BaseModel):
    web_search : bool 
    vectorstore : bool
    get_weather :bool
structured_router_model = llm.with_structured_output(RouterModel)
router_system_prompt2 = """
You are supposed to return with the tools which can be used in the user input , note that multiple tools can be used 
vectorstore tools is used for when the user input is related to premium policy , get_weather is used when the user asks for the weather of a particular city. Use web_search for any other query
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system_prompt2),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_router_model


#RAG CHAIN DONE
prompt = hub.pull("rlm/rag-prompt" , api_key = langchain_api_key)
rag_chain = prompt | llm | StrOutputParser()

class RAGPromptEnhancerModel(BaseModel):
    rag_enhanced_query: str

structured_rag_prompt_model = llm.with_structured_output(RAGPromptEnhancerModel)
rag_enhanced_query = """
You are an expert assistant for extracting queries related to Retrieval-Augmented Generation (RAG).
The vectorstore contains documents related to motor insurance policies, which include:

1. General Conditions: Definitions, governing clauses.
2. Compulsory Civil Liability Insurance: Coverage for bodily injuries and material damages to third parties.
3. Exclusions: Situations excluded from coverage.
4. Premium Payments and Alterations: Payment schedules, consequences of non-payment, adjustments.
5. Risk and Obligations: Risk assessment, claims reporting, and risk mitigation duties.
6. Optional Covers: Extended guarantees, exclusions for improper use or unlicensed drivers.
7. Specific Scenarios and Clauses: Coverage for garage owners, driving instruction, and more.
8. Bonuses and Penalties: Incentives for no-claims behavior and penalties for frequent claims.
9. Jurisdiction and Dispute Resolution: Legal frameworks for dispute resolution.
10. Miscellaneous Provisions: Proof of insurance, special conditions, and document submission requirements.

Your task:
- Extract queries specifically related to motor insurance, insurance policies, premiums, coverage, exclusions, risk assessment, or legal frameworks.
- Ignore any queries or details related to weather functions or queries unrelated to insurance policies.
- Ensure the query is clear, concise, and directly related to any of the listed topics.
- If the query includes non-relevant information (like weather-related details), filter those out and focus only on insurance, policy, and premium-related topics.
"""
rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_enhanced_query),
        ("human", "{question}"),
    ]
)
rag_prompt_enhancer = rag_prompt_template | structured_rag_prompt_model


#WEATHER NAME DONE 

class cityname(BaseModel):
    city_name : str = Field(
        ...,
        description="The name of the city",
    )

class WeatherResponse(BaseModel):
    weather_output: str = Field(
        ...,
        description = "Weather response of the user input"
    )
structuctured_cityname_model = llm.with_structured_output(cityname)
structured_weather_output = llm.with_structured_output(WeatherResponse)
with open("prompts/weather_prompt.md", "r") as file:
    weather_system_prompt = file.read()
city_name_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", weather_system_prompt),
        ("human", "Here is the user input : \n\n {text_input}"),
    ]
)
with open("prompts/weather_response_prompt.md" , "r") as file:
    weather_response_system_prompt = file.read()
weather_response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", weather_response_system_prompt),
        ("human", "Here is the user input : \n\n {text_input} , Here is the details for the {city_name} : \n\n Temperature : {temperature} , Feels like temperature : {feels_like} , Maximum temperature of the day : {temp_max} , Minimum temperature of the day : {temp_min}, The current date and time for that city : {local_time}"),
    ]
)
city_extractor = city_name_prompt | structuctured_cityname_model
weather_response = weather_response_prompt | structured_weather_output
def fetch_weather(text_input): 
    print("FETCH WEATHER FUNCTION INVOKED")
    city_name = city_extractor.invoke({"text_input": text_input}).city_name
    print(city_name)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={openweather_api_key}&units=metric"
    response = requests.get(url)
    response.raise_for_status()  
    weather_info = response.json()
    temperature = weather_info['main']['temp']
    feels_like = weather_info['main']['feels_like']
    temp_min = weather_info['main']['temp_min']
    temp_max = weather_info['main']['temp_max']
    weather_description = weather_info['weather'][0]['description'].capitalize()
    humidity = weather_info['main']['humidity']
    wind_speed = weather_info['wind']['speed']
    timezone_offset_seconds = weather_info['timezone']
    tz = timezone(timedelta(seconds=timezone_offset_seconds))
    current_utc_time = datetime.now(timezone.utc)
    local_time = current_utc_time.astimezone(tz)
    formatted_time = local_time.strftime('%d-%m-%Y %H:%M:%S')
    result = (
        f"Weather Report for {city_name}:\n"
        f"Date & Time: {formatted_time}\n"
        f"Description: {weather_description}\n"
        f"Temperature: {temperature}°C\n"
        f"Feels Like: {feels_like}°C\n"
        f"Min Temperature: {temp_min}°C\n"
        f"Max Temperature: {temp_max}°C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind_speed} m/s\n"
    )
    return str(result)
web_search_tool  = TavilySearchResults(k=5)

# WEB SEARCH QUERY ENHANCER TOOL 
class WebSearchEnhancerModel(BaseModel):
    extracted_query: str

structured_web_search_model = llm.with_structured_output(WebSearchEnhancerModel)

# Define the Web Search system prompt
web_search_prompt = """
You are an assistant tasked with extracting the web search-related part of a query. 
The input query is already determined to require web search, so your job is to extract only the relevant part of the query for performing the search.
Note : You are strictly supposed to ignore the part of query related to policies and their premiums and the weather of cities and countries
Rules:
IMPORTANT NOTE : You are supposed to ignore the part of query where the user asks about the weather of a particular city
1. Ignore any parts of the query related to RAG (Retrieval-Augmented Generation) or vectorstore content, such as motor insurance policies.
2. Ignore parts of the query asking about the weather of a particular city.
3. Extract only the web search-related content, concisely and precisely.
"""
web_search_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", web_search_prompt),
        ("human", "{question}"),
    ]
)

web_search_enhancer = web_search_prompt_template | structured_web_search_model


#output combiner 
class OutputCombinerModel(BaseModel):
    combined_output: str

# Initialize the LLM with structured output
structured_combiner_model = llm.with_structured_output(OutputCombinerModel)

# Define the Output Combiner system prompt
output_combiner_prompt = """
combine the outputs given as input with the user question , return the restructured combined output on the user tone
"""

output_combiner_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", output_combiner_prompt),
        ("human", "User input: {user_input}, Output1: {output1}, Output2: {output2}, Output3: {output3}"),
    ]
)

# Combining outputs and user input
output_combiner = output_combiner_prompt_template | structured_combiner_model

def combine_outputs(user_input, output1, output2, output3):
    input_data = {
        "user_input": user_input,
        "output1": output1,
        "output2": output2,
        "output3": output3
    }

    # Use the output_combiner to process the inputs
    combined_output = output_combiner.invoke(input_data)

    # Return the final result
    return combined_output.combined_output
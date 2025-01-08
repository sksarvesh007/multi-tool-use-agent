from chunk_extractor import chunker_extractor
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import dotenv
dotenv.load_dotenv()

#loading chunks from pdf
knowledge_pdf = "general_policy.pdf"
chunks = chunker_extractor(knowledge_pdf)

#loading the embedding model
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)

#making index in pinecone DB 

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "agentic-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

time.sleep(5)

# See how many vectors have been upserted
print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
time.sleep(2)
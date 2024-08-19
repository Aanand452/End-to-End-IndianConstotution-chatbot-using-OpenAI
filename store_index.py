import os
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI, OpenAIEmbeddings
from src.helper import load_pdf, text_splitter
from dotenv import load_dotenv

#import dotenv

load_dotenv()

# Retrieve the OpenAI API key from environment variables

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Initialize the OpenAI client
client = OpenAI(openai_api_key=OPENAI_API_KEY)

# print(client)

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)




#import the pinecone and environment from the dotenv
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# print(PINECONE_API_KEY)
# print(PINECONE_ENV )

extracted_data =load_pdf("pdfs/")
text_chunks = text_splitter(extracted_data)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


#initiate the connection to connect Pinecone 

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY,environment=PINECONE_ENV)

# define the index_name 
index_name = "constitutionthree"

#create the index_name if not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
index = pc.Index(index_name)


vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings.embed_query ,
    index_name=index_name
)

docSearch=vector_store.from_texts([t.page_content for t in text_chunks],embedding=embeddings,index_name=index_name)




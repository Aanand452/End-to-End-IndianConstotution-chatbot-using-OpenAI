import os 
from dotenv import load_dotenv
from flask import Flask,render_template,jsonify,request
from langchain_openai import OpenAI, OpenAIEmbeddings
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from store_index import *
from src.helper import *
from src.prompt import *
from langchain import PromptTemplate


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Initialize the OpenAI client
client = OpenAI(openai_api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY,environment=PINECONE_ENV)

# index_name = "constitutionthree"

    
# index = pc.Index(index_name)

# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embeddings.embed_query ,
#     index_name=index_name
# )




#loading the index

docSearch = vector_store.from_existing_index(index_name,embeddings)


prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt":prompt}

qa = RetrievalQA.from_chain_type(
    llm=client,
    chain_type="stuff",
    retriever=docSearch.as_retriever(),
    return_source_documents=True
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query":input})
    print("Response :",result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug=True)
               


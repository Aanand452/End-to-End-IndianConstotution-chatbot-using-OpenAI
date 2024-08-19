from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings 


#Extract the data from the pdf

def load_pdf(pdfs):
    loader=PyPDFDirectoryLoader("pdfs")
    
    documents =loader.load()
    
    return documents

# extracted_data =load_pdf("pdf/")

# print(extracted_data)

#create the chunks 

def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks


#create the embeedings 

# embeddings = OpenAIEmbeddings(openai_api_type=OPENAI_API_KEY)


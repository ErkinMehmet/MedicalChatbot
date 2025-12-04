from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
def extract_data_pdf(path):
    loader=DirectoryLoader(path,glob="*.pdf",loader_cls=PyMuPDFLoader)
    documents=loader.load()
    return documents

def  text_split(pdf_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts=text_splitter.split_documents(pdf_data)
    return texts   

def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
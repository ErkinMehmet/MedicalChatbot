from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import List
from langchain.schema import Document

def extract_data_pdf(path):
    loader=DirectoryLoader(path,glob="*.pdf",loader_cls=PyMuPDFLoader)
    documents=loader.load()
    return documents

def  text_split(pdf_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts=text_splitter.split_documents(pdf_data)
    return texts   

def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get('source','')
        minimal_docs.append(Document(page_content=doc.page_content,metadata={'source':src}))
    return minimal_docs
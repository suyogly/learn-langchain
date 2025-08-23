# this is the traditional keyword matching retrieval method called BM25

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# load the document

pdf = PyPDFLoader(file_path="files/jim-collins-beyond-entrepreneurship.pdf")
doc = pdf.load()
# print(doc)

chunk = BM25Retriever(

)
# this is the traditional keyword matching retrieval method called BM25

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma (as we dont use vector stores in sparse retrieval)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# load the document

pdf = PyPDFLoader(file_path="files/jim-collins-beyond-entrepreneurship.pdf")
doc = pdf.load()
# print(doc)

# chunk the document with text splitters
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

splitted_content = splitter.split_documents(doc)
# print(splitted_content)

# define retriever(BM25)
retriever = BM25Retriever.from_documents(
    documents=splitted_content,
    k=2
)

retriever.invoke()
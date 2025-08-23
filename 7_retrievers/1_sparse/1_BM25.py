# this is the traditional keyword matching retrieval method called BM25

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma (as we dont use vector stores in sparse retrieval)
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

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
    k=4
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=None)

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:
""")

parser = StrOutputParser()

def join_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


chain = (
    {
        "context": retriever | join_doc,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

result = chain.invoke("how to get rich")

print(result)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader(file_path="files/Reality transurfing Steps I-V - PDF Room.pdf")

content = loader.load()

# print(content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 270
)

splitted_content = splitter.split_documents(content)

# print(splitted_content)

vector_db = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    collection_name="dense"
)

vectors = vector_db.add_documents(splitted_content)
# print(vectors)

retriever = vector_db.as_retriever(
    search_type = "similarity",
    search_kwargs={"k": 4}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=None
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:
""")

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

result = chain.invoke("what is the essence of this book? in an utmost detail")

print(result)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

file = PyPDFLoader(file_path="files/Reality transurfing Steps I-V - PDF Room.pdf")
content = file.load()
# print(content)

chunks = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 250
)

embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = Chroma(
    embedding_function=embed,
    persist_directory="new_db",
    collection_name="dense"
)

print(vector)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

file = TextLoader(file_path="files/hello.txt")
content = file.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)

split = splitter.split_documents(documents=content)

print(split)
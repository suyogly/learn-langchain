from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

doc1 = PyPDFLoader(file_path="files/jim-collins-beyond-entrepreneurship.pdf")
doc2 = TextLoader(file_path="files/hello.txt")
doc3 = CSVLoader(file_path="files/hello.csv")

doc1_content = doc1.load()
doc2_content = doc2.load()
doc3_content = doc3.load()

splitter = CharacterTextSplitter(
    chunk_size=50, # chunk with how many characters?
    chunk_overlap=0, # from the last chunk to present chunk, how many of the chunk will overlap?
    separator=''
)

split1 = splitter.split_documents(documents=doc1_content)
split2 = splitter.split_documents(documents=doc2_content)
split3 = splitter.split_documents(documents=doc3_content)

print(f"Document 1, .pdf file: \n {split1[0].page_content} \n")
print(f"Document 2, .txt file: \n {split2[0].page_content} \n")
print(f"Document 3, .csv file: \n {split3[0].page_content} \n")


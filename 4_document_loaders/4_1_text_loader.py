from langchain_community.document_loaders import TextLoader

loader = TextLoader("files/hello.txt")

result = loader.load()
print(result)

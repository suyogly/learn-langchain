from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

# load text files
txt_loader = TextLoader("files/hello.txt")

text_content = txt_loader.load()
print(text_content)
print("\n ----------------- \n")

prompt1 = PromptTemplate(
    template="check what's there in the {text}, and tell me the gist about it.",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | llm1 | parser

result = chain.invoke({"text" : text_content})
print(result)
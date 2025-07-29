from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

csv = CSVLoader(file_path="files/hello.csv")

csv_content = csv.load()

print(len(csv_content))

print("\n ----- \n")

prompt1 = PromptTemplate(
    template="generate the historic and contemporary short analysis of {data} from {csv}",
    input_variables= ["data", "csv"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser

result = chain.invoke({
    "data" : "history of jazz music",
    "csv" : csv_content
})

print(result)
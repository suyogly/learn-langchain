from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

prompt1 = PromptTemplate(
    template="generate the 10 point summary of {country} political crisis",
    input_variables=["country"]
)

prompt2 = PromptTemplate(
    template="Describe in details about each points from {response}",
    input_variables=["response"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({"Nepal"})
print(result)
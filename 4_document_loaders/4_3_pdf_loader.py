from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

pdf = PyPDFLoader(file_path="files/jim-collins-beyond-entrepreneurship.pdf")
content = pdf.load()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=None
)

prompt = PromptTemplate(
    template="from the {pdf}, please let me know about this {topic}",
    input_variables=["pdf", "topic"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "pdf" : content,
    "topic": "what does he talk about in chapter 2"
})

print(result)

'''
But what about unstructured pdf? 

we can use unstructuredpdfloader package, or can use text extractors like amazon textract

'''
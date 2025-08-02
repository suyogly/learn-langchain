from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

'''
WebBaseLoader for static pages - bs4 is required
SeleniumURLLoader for JS heavy sites - selenium is required

'''

site = WebBaseLoader("https://suyogly.com/blog/on-learnings-ai-and-sharings/")

content = site.load()
print(content)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=None
)

prompt = PromptTemplate(
    template="please tell me what {url} is about about",
    input_variables=["url"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "url" : content
})

print(result)
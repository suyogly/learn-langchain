from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

search = TavilySearchResults(kwargs=4)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=None)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Based on these search results about '{query}', provide a summary:\n\n{search_results}"
)

chain = {
    "query": RunnablePassthrough(),
    "search_results": search
} | prompt | llm | parser

result = chain.invoke("social media bans in nepal")
print(result)

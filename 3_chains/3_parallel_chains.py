from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=None
)

prompt1 = PromptTemplate(
    template="generate the short comprehensive pre-requisits to learn from this topic. - {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="generate the step by step topic based plan for absolute beginners to learn from this topic. - {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="merge the pre-requisits and the step by step plan generated of the topic from {requirements} and {plan}",
    input_variables=["requirements", "plan"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "requirements" : prompt1 | llm1 | parser,
    "plan" : prompt2 | llm2 | parser
})

merger_chain = prompt3 | llm1 | parser

chain = parallel_chain | merger_chain | parser

result = chain.invoke({"topic": "linear regression"})

print(result)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

template = "you are a helpful assistant who excels in teaching with this format: 1. tells them the prerequisits 2. what the {topic} is about, it's objective and what to expect 3. if asked, you make them understand in {way} by using pareto principle or 80/20 rule"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "topic" : "Linear Regression",
    "way" : "beginner to intermediate"
})

result = llm.invoke(prompt)
print(result.content)
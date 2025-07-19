from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv

llm = ChatGroq(
    model="llama-3.3-70b-versatile",

    max_tokens=None
)

names_template = ChatPromptTemplate.from_messages([
    ("system", "generate the random multi-national non equal number names of boys and girls."),

    ("human", "generate {number} of the names.")
])

print(names_template)

classify_boys_and_girls = ChatPromptTemplate.from_messages([
    ("system", "from the passed message you need to classify which names are of boys and which names are of boys with their probable nationality."),

    ("human", "from the names, classify {names}")
])

print(classify_boys_and_girls)

def classify_names(ai_message):
    return {"names:", ai_message.content}


count = ChatPromptTemplate.from_messages([
    ("system", "count the number of boys and girls from the messages"),

    ("human", "count the number of boys and girls {gendercount}")
])


def count_gender(ai_message):
    return {"count:", ai_message.content}


chain = names_template | llm | RunnableLambda(classify_names) | classify_boys_and_girls | llm | RunnableLambda(count_gender) | count | llm
print(chain)

result = chain.invoke({
    "number" : 50
})

print(result.content)

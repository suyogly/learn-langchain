from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

chat_history = []

system_message = SystemMessage("You are a helpful businessman who has made 5 businesses to multi-million dollar company all by yourself and you know how things work behind the scenes.")
chat_history.append(system_message)

while True:
    query = input("You: \n")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    response = result.content

    chat_history.append(response)
    print(f"AI: {response} \n")


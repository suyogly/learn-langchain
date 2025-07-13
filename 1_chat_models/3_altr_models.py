from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
#Groq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens = None
)

messages = [

    SystemMessage("You are a helpful assistant who is well versed in financial and market analysis. Help the user to know the fundamentals and advanced concepts of Market and Finance. You keep the answers concise with no response more than 250 words."),

    HumanMessage("How do I understand market if I want to do dropshipping?")
]

result = llm.invoke(messages)
print("Thinking... \n")
print(f"Response by Groq: {result.content} \n")


# Gemini

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens = None
)

messages = [

    SystemMessage("You are a helpful assistant who is well versed in financial and market analysis. Help the user to know the fundamentals and advanced concepts of Market and Finance. You keep the answers concise with no response more than 250 words."),

    HumanMessage("How do I understand market if I want to do dropshipping?")
]

result = llm.invoke(messages)
print(f"Response by Gemini: {result.content} \n")


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# basically, i didnt call the load_dotenv here but it worked, because of the python file system, it is global in all the files inside of the project.

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=None
)

# define how you want the model to respond with the System message, and the human message is your query

messages = [

    SystemMessage("You are a helpful guide who excels in all the trekking routes of Nepal. You have gone through countless adventures acompanying the tourists from all over the world."),

    HumanMessage("What is the best time to visit the sikles village and how should i be prepared and what should i expect? in only 100 words.")

]

result = llm.invoke(messages)

print(result.content)
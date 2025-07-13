# import the chatmodel you want to use, here i am importing gemini
# import dotenv to store api in .env file

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# which model to use, we can pass arguments like how much token to use, randomness of the response (temperature)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=500
)

result = llm.invoke("tell me about Nepal.")

# this prints all the result from answers to token consumptions
print(result)

# this prints only the response
print(result.content)
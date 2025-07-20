from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# what we are going to do is to create a small summary, main lesson and where we can apply that lesson from a story

llm1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=None
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=None
)

llm3 = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=None
)

prompt1 = PromptTemplate(
    template="generate a small summary expressing from start to finish, containing main happenings from this {story} in flat string only",
    input_variables=["story"]
)

prompt2 = PromptTemplate(
    template="covey the main lesson this {story} is sharing. the gist of the lesson.",
    input_variables=["story"]
)

prompt3 = PromptTemplate(
    template="merge the summary and the main lesson of the story from {summary} and {lessons}, and also mention how the lesson can be applied in real life with example.",
    input_variables=["summary", "lessons"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "summary" : prompt1 | llm1 | parser,
    "lessons" : prompt2 | llm2 | parser
})

merger_chain = prompt3 | llm3 | parser

chain = parallel_chain | merger_chain | parser

story1 = """ One day a farmers donkey fell down into a well. The animal cried piteously for hours as the farmer tried to figure out what to do. Finally, he decided the animal was old, and the well needed to be covered up anyway-it just wasn’t worth it to retrieve the donkey.

He invited all of his neighbors to come over and help him. They all grabbed a shovel and began to shovel dirt into the well. At first, the donkey realized what was happening and cried horribly. Then, to everyone’s amazement, he quieted down.

A few shovel loads later, the farmer finally looked down the well. He was astonished at what he saw. With each shovel of dirt that hit his back, the donkey was doing something amazing. He would shake it off and take a step up.

As the farmers neighbors continued to shovel dirt on top of the animal, he would shake it off and take a step up.

Pretty soon, everyone was amazed as the donkey stepped up over the edge of the well and happily trotted off! """

result = chain.invoke({"story": story1})

print(result)
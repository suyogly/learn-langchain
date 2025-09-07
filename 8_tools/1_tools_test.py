from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
search_result = search.invoke("Capital of Nepal")
print(search_result)
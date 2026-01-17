from typing import Any, Dict
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from graphs.state import GraphState # Use your defined state class

web_search_tool = TavilySearch(max_results=5)

def web_search_node(state: GraphState) -> Dict[str, Any]:
    #print("---NODE: WEB SEARCH---")

    messages = state.get("messages", [])
    
    if not messages:
        # Fallback for the very first run if 'messages' isn't populated
        # This usually only happens if the input dict is wrong
        last_user_message = state.get("question", "No question provided")
    else:
        last_user_message = messages[-1].content
    
    # 1. Fetch from Tavily
    search_results = web_search_tool.invoke({"query": last_user_message})
    
    # 2. Format into a clean string
    # We include the URL so the AI can reference where the info came from
    web_results_str = "\n\n".join(
        [f"Source: {d['url']}\nContent: {d['content']}" for d in search_results]
    )

    # 3. Update the State using the correct key
    return {"research_context": web_results_str}
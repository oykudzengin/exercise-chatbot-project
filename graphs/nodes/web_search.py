from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from graphs.state import GraphState # Use your defined state class

web_search_tool = TavilySearchResults(k=3)

def web_search_node(state: GraphState) -> Dict[str, Any]:
    print("---NODE: WEB SEARCH---")

    question = state["question"]
    
    # 1. Fetch from Tavily
    search_results = web_search_tool.invoke({"query": question})
    
    # 2. Format into a clean string
    # We include the URL so the AI can reference where the info came from
    web_results_str = "\n\n".join(
        [f"Source: {d['url']}\nContent: {d['content']}" for d in search_results]
    )

    # 3. Update the State using the correct key
    return {"research_context": web_results_str}
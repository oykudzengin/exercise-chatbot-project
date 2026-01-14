from typing import Any, Dict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from state import GraphState

web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---NODE: WEB SEARCH---")

    question = state["question"]
    # Ensure this matches the key in your state.py (e.g., "documents" or "research_context")
    existing_docs = state.get("documents", [])

    # 1. Fetch from Tavily
    search_results = web_search_tool.invoke({"query": question})
    
    # 2. Format into a clean string for the LLM
    web_results_str = "\n\n".join(
        [f"Source: {d['url']}\nContent: {d['content']}" for d in search_results]
    )

    # 3. Create the Document object
    web_doc = Document(
        page_content=web_results_str,
        metadata={"source": "tavily_web_search"}
    )

    # 4. Append to the list
    if existing_docs is None:
        updated_docs = [web_doc]
    else:
        # We use a list copy to avoid mutating state directly
        updated_docs = list(existing_docs) 
        updated_docs.append(web_doc)

    return {"research_context": web_results_str}
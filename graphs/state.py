from typing import List, TypedDict, Optional

class GraphState(TypedDict):
    """Represents the state of our graph"""
    
    question: str              # User's input question
    user_profile: Optional[dict] # Extracted user profile
    
    safe_exercises: List[dict]   # Filtered list from JSON
    research_context: str        # Relevant text chunks from Pinecone
    web_search_results: str      # Data from Tavily (optional)

    datasource: str              # 'local_db' or 'web_search'
    is_safe: str                 # "yes" or "no" for safety_grader
    explanation: str             # The feedback from the grader
    loop_count: int              # To track how many times we've retried

    generation: str              # Final answer given by the model
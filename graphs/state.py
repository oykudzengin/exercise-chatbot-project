from typing import List, TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    """Represents the state of our graph"""

    #we now have messages instead of the question
    #question: str              # User's input question

    #new messages will be appended to the list
    messages: Annotated[List[BaseMessage], operator.add]

    user_profile: Optional[dict] # Extracted user profile
    
    safe_exercises: List[dict]   # Filtered list from JSON
    onboarding_complete: bool
    research_context: str        # Relevant text chunks from Pinecone
    web_search_results: str      # Data from Tavily (optional)

    datasource: str              # 'local_db' or 'web_search'
    is_safe: str                 # "yes" or "no" for safety_grader
    explanation: str             # The feedback from the grader
    loop_count: int              # To track how many times we've retried

    generation: str              # Final answer given by the model
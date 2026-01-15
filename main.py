import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from graphs.state import GraphState
from graphs.nodes.query_analysis import query_analyzer_node
from graphs.nodes.retriever import retriever_node
from graphs.nodes.web_search import web_search
from graphs.nodes.generator import generator_node
from graphs.nodes.safety_grader import safety_grader_node

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def route_question(state: GraphState):
    """
    Determines if we go to the local VectorDB or the Web.
    """
    if state.get("datasource") == "web_search":
        print("---ROUTING TO WEB SEARCH---")
        return "web_search"
    elif state.get("datasource") == "none":
        print("---ROUTING TO GENERAL CHAT---")
        return "just_chat"
    else:
        print("---ROUTING TO LOCAL RETRIEVER---")
        return "retrieve_local"
    
def check_safety_results(state: GraphState):
    """
    This function acts as the 'Police Officer' at the crossroads.
    """
    # If the grader says it's safe OR we've tried too many times (e.g., 3)
    if state.get("is_safe") == "yes" or state.get("loop_count", 0) >= 3:
        return "finish"
    
    # If it's unsafe and we have retries left, go back to generator
    print(f"SAFETY CHECK FAILED: {state.get('explanation')}. Retrying...")
    return "retry"

def route_after_generation(state: GraphState):
    """
    Decide whether to grade the output or just finish.
    """
    # If the intent was a greeting or general chat, just stop.
    # We access this from the user_profile we stored in query_analysis.
    user_profile = state.get("user_profile", {})
    if user_profile.get("intent") in ["greeting", "general_chat"]:
        return "finish"
    
    # Otherwise, it's a workout—run it by the doctor (grader)
    return "grade"

workflow = StateGraph(GraphState)
memory = InMemorySaver()

#addin the nodes created
workflow.add_node("analyze_query", query_analyzer_node)
workflow.add_node("retrieve_local", retriever_node)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_workout", generator_node)
workflow.add_node("safety_grader", safety_grader_node)

# Add Edges (Hallways)
workflow.add_edge(START, "analyze_query")

workflow.add_conditional_edges(
    "analyze_query",
    route_question,
    {
        "retrieve_local": "retrieve_local",
        "web_search": "web_search",
        "just_chat": "generate_workout"
    }
)

workflow.add_edge("web_search", "generate_workout")
workflow.add_edge("retrieve_local", "generate_workout")

# The Safety Loop: Generator -> Grader -> (Finish or Back to Generator)
workflow.add_conditional_edges(
    "generate_workout",
    route_after_generation,
    {
        "finish": END,
        "grade": "safety_grader"
    }
)

workflow.add_conditional_edges(
    "safety_grader",
    check_safety_results,
    {
        "finish": END,
        "retry": "generate_workout" 
    }
)

# Compile
app = workflow.compile(checkpointer=memory)

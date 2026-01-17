import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from graphs.state import GraphState
from graphs.nodes.query_analysis import query_analyzer_node
from graphs.nodes.retriever import retriever_node
from graphs.nodes.web_search import web_search_node
from graphs.nodes.generator import generator_node
from graphs.nodes.safety_grader import safety_grader_node
from graphs.nodes.greeting import greeting_node
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def route_greeting(state: GraphState):
    """
    If onboarding is not complete (first message), we wait for user input.
    Otherwise, we proceed to analysis.
    """
    if state.get("onboarding_complete") is False:
        print("---ONBOARDING INCOMPLETE: WAITING FOR USER---")
        return "wait_for_user"
    print("---ONBOARDING COMPLETE: PROCEEDING TO ANALYSIS---")
    return "analyze"

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

def get_graph():
    workflow = StateGraph(GraphState)

    #adding the nodes created
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("analyze_query", query_analyzer_node)
    workflow.add_node("retrieve_local", retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_workout", generator_node)
    workflow.add_node("safety_grader", safety_grader_node)

    #adding the edges
    # 1. Start with Greeting
    workflow.add_edge(START, "greeting")

    # 2. Conditional Edge from Greeting
    workflow.add_conditional_edges(
        "greeting",
        route_greeting,
        {
            "wait_for_user": END,   # Stops here so user can see the greeting message and reply
            "analyze": "analyze_query" # Proceeds if user has already replied
        }
    )
    # 3. After analyzing the query it moves to corresponding node
    workflow.add_conditional_edges(
        "analyze_query",
        route_question,
        {
            "retrieve_local": "retrieve_local",
            "web_search": "web_search",
            "just_chat": "generate_workout"
        }
    )
    # 4. Edges leading to generator node
    workflow.add_edge("web_search", "generate_workout")
    workflow.add_edge("retrieve_local", "generate_workout")

    # The Safety Loop: Generator -> Grader -> (Finish or Back to Generator)
    workflow.add_conditional_edges(
        "generate_workout",
        route_after_generation,
        {
            "finish": END,
            "grade": END
            #"grade": "safety_grader" grader not is temporarily on hold!!!
        }   
    
    )

    # Depending on the safety results either generate again or give the final answer
    workflow.add_conditional_edges(
        "safety_grader",
        check_safety_results,
        {
            "finish": END,
            "retry": "generate_workout" 
        }
    )

    memory = InMemorySaver()
    # Compile
    app = workflow.compile(checkpointer=memory)
    return app

ex_chatbot_app = get_graph()













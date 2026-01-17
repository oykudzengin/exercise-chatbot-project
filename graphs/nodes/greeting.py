from langchain_core.messages import AIMessage
from graphs.state import GraphState

def greeting_node(state):
    # Check if this is the very first message
    if not state.get("messages") or len(state.get("messages")) == 0:
        greeting = (
            "Welcome to Elite Medical Fitness! 🏋️‍♂️\n\n"
            "To build a workout that is safe for your body, I need to know three things:\n"
            "1. Your **Experience Level** (Beginner, Intermediate, or Advanced)\n"
            "2. Your **Goal** for today (Lower Body, Upper Body, Cardio, or a specific muscle)\n"
            "3. Any **Health Conditions** or pains (e.g., 'Lower back pain', 'Hypertension', or 'None')\n\n"
            "How can I help you get started today?"
        )
        return {"messages": [AIMessage(content=greeting)], "onboarding_complete": False}
    
    # If messages exist, we just pass through to the analyzer
    return {"onboarding_complete": True}
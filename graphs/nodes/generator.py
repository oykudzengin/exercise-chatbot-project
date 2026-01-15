from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

def generator_node(state):
    print("---NODE: GENERATING WORKOUT PLAN---")

    history = state.get("messages", [])
    exercises = state.get("safe_exercises", [])
    research = state.get("research_context", "No additional research available.")
    user = state.get("user_profile", {})

    current_loops = state.get("loop_count", 0)
    feedback = state.get("explanation", "None")

    # If feedback exists, the AI will prioritize fixing that mistake
    if feedback != "None" and feedback != "Passed":
        safety_msg = f"\n\n CRITICAL: Your previous plan was REJECTED for safety reasons: {feedback}. You MUST fix this in the new version."
    else:
        safety_msg = ""
    
    #Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an Elite Medical Fitness Coach. I will provide you with a list of 'Safe Exercises' from our database.
        If the list has exercises, prioritize them.
        If the list is empty, use your internal knowledge to suggest the safest possible bodyweight exercises for the user's condition. Never tell the user 'no exercises were provided'; simply provide the best possible advice.
        Use the Research Context to explain WHY these exercises are safe and what to avoid.{safety_msg}

        If no 'Safe Exercises' are provided, it means the user is just greeting you or asking a general question. Respond naturally to their message and encourage them to share their fitness goals or any physical conditions so you can create a safe plan for them.
        
        Format:
        - Brief encouragement
        - Safety Precautions (based on research)
        - The Workout (list exercises, sets, and reps)
        - 'When to Stop' red flags.

        Refer to the user's history to see if they are asking for changes to a previous plan."""),
        ("placeholder", "{chat_history}"),
        ("human", "User Profile: {user}\nSafe Exercises: {exercises}\nResearch Context: {research}")
    ])
    
    #Initialize Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    chain = prompt | llm | StrOutputParser()
    
    #Run and update state
    response = chain.invoke({
        "chat_history": history,
        "user": user, 
        "exercises": exercises, 
        "research": research
    })

    # If your chain has a StrOutputParser(), 'response' is a string.
    # We need to turn it into an AIMessage for the messages list.
    if isinstance(response, str):
        content = response
        message = AIMessage(content=content)
    else:
        content = response.content
        message = response
    
    return {
        "messages": [message],
        "generation": content,
        "loop_count": current_loops + 1
        }
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generator_node(state):
    print("---NODE: GENERATING WORKOUT PLAN---")

    exercises = state["safe_exercises"]
    research = state["research_context"]
    user = state["user_profile"]

    current_loops = state.get("loop_count", 0)
    feedback = state.get("explanation", "None")

    # If feedback exists, the AI will prioritize fixing that mistake
    if feedback != "None" and feedback != "Passed":
        safety_msg = f"\n\n CRITICAL: Your previous plan was REJECTED for safety reasons: {feedback}. You MUST fix this in the new version."
    else:
        safety_msg = ""
    
    #Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an elite Medical Fitness Coach. 
        Create a workout plan based ONLY on the safe exercises provided.
        Use the Research Context to explain WHY these exercises are safe and what to avoid.{safety_msg}
        
        Format:
        - Brief encouragement
        - Safety Precautions (based on research)
        - The Workout (list exercises, sets, and reps)
        - 'When to Stop' red flags.
        """),
        ("human", "User Profile: {user}\nSafe Exercises: {exercises}\nResearch Context: {research}")
    ])
    
    #Initialize Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    chain = prompt | llm | StrOutputParser()
    
    #Run and update state
    response = chain.invoke({
        "user": user, 
        "exercises": exercises, 
        "research": research
    })
    
    return {
        "generation": response,
        "loop_count": current_loops + 1
        }
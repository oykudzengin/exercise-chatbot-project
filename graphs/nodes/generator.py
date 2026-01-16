from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
import os



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
        ("system", """You are a Medical Fitness Coach. You must synthesize the provided data to guide the user.
            DATA SOURCES PROVIDED:
            1. SAFE EXERCISES: {exercises} (The only exercises you are allowed to recommend).
            2. RESEARCH CONTEXT: {research} (Clinical guidelines from our medical PDFs).
            3. USER PROFILE: {user} (The patient's current status).

            INSTRUCTIONS:
            - INTEGRATION: Cross-reference the USER PROFILE with the RESEARCH CONTEXT. If the research mentions precautions for the user's specific conditions (e.g., hypertension, lower back pain), apply those filters to the exercise selection.
            - COACH'S TIP: Provide a 'clinical' tip for every exercise. Use the RESEARCH CONTEXT to explain the physiological benefit (e.g., 'Maintaining a neutral spine reduces intradiscal pressure as noted in our protocols').
            - FORMATTING: Use Markdown tables for the workout plan.
            - FALLBACK: If the 'SAFE EXERCISES' list is empty or matches the user's 'not_suitable_for' list, do NOT suggest random exercises. Instead, provide 3-4 'pre-habilitation' or lifestyle tips found in the RESEARCH CONTEXT."""),
        ("placeholder", "{chat_history}"),
        ("human", "User Profile: {user}\nSafe Exercises: {exercises}\nResearch Context: {research}\n{safety_feedback}")
    ])
    
    #GENERATOR_GOOGLE_API_KEY = os.getenv("GENERATOR_GOOGLE_API_KEY")
    GRADER_API_KEY = os.getenv("GRADER_API_KEY")
    #Initialize Chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GRADER_API_KEY
        )
    chain = prompt | llm | StrOutputParser()
    
    #Run and update state
    response = chain.invoke({
        "chat_history": history,
        "user": user, 
        "exercises": exercises, 
        "research": research,
        "safety_feedback": safety_msg
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
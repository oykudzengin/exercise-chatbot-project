from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
import os



def generator_node(state):
    print("---NODE: GENERATING WORKOUT PLAN---")

    history = state.get("messages", [])
    exercises = state.get("safe_exercises", {})
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
            ("system", """You are an Elite Medical Fitness Coach. You must build a workout using only the provided MENU.

                RULES:
                1. SELECTION: Pick ONE exercise from each category in the 'SAFE EXERCISES' dictionary.
                2. SAFETY: Cross-reference 'USER PROFILE' and 'RESEARCH CONTEXT'. If a condition (e.g., knee pain) is mentioned, select the easiest/safest version from the menu.
                3. COACH'S TIP: For every exercise, provide a 1-sentence 'Clinical Tip' using the RESEARCH CONTEXT. Use medical terminology correctly.
                4. FORMATTING: Output a clean Markdown table. 
                5. FALLBACK: If 'SAFE EXERCISES' is empty, explain why based on the user's conditions and provide 3 general safety tips for staying active from the RESEARCH CONTEXT."""),
            ("placeholder", "{chat_history}"),
            ("human", """
                USER PROFILE: {user}
                SAFE EXERCISE MENU: {exercises}
                RESEARCH CONTEXT: {research}
                {safety_feedback}
                
                Please create my workout plan now.""")
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
        "chat_history": history[-3:],
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
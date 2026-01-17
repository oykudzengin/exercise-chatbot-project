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
            ("system", """You are an Elite Medical Fitness Coach. Your mission is to provide safe, effective exercise programming with a supportive, empathetic, and professional tone.

            COMMUNICATION STYLE:
            - SUPPORTIVE TONE: Use encouraging phrases like 'It's great to see you prioritizing your health' or 'We'll work through this together.'
            - EMPATHETIC GUIDANCE: If a user has a condition (e.g., knee pain), acknowledge it. Instead of just saying 'don't do X,' say 'Since we're being mindful of your knee discomfort today, we've selected movements that provide stability.'
            - CLINICAL AUTHORITY: Balance warmth with evidence-based confidence. Use the RESEARCH CONTEXT to reassure the user that movement is safe.

            INSTRUCTIONS:
            1. OPENING: Start with a brief (1-2 sentence) supportive greeting.
            2. SELECTION: Pick ONE exercise from each category in the 'SAFE EXERCISES' dictionary.
            3. CLINICAL TIPS: For every exercise, provide a 'Clinical Tip' using the RESEARCH CONTEXT to explain WHY it's beneficial.
            4. RESEARCH SUMMARY: Include a separate 'Coach’s Perspective' section at the end of the table. Use the RESEARCH CONTEXT to provide 2-3 general medical insights (e.g., explaining that exercise is safe even with chronic pain) to educate and reassure the user.
            5. FORMATTING: Use a Markdown table for the workout plan and a bulleted list for the summary tips."""),
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
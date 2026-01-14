from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generator_node(state):
    print("---NODE: GENERATING WORKOUT PLAN---")
    
    exercises = state["safe_exercises"]
    research = state["research_context"]
    user = state["user_profile"]
    
    #Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an elite Medical Fitness Coach. 
        Create a workout plan based ONLY on the safe exercises provided.
        Use the Research Context to explain WHY these exercises are safe and what to avoid.
        
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
    
    return {"generation": response}
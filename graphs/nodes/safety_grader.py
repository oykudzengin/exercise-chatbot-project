from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

GRADER_API_KEY = os.getenv("GRADER_API_KEY")
class SafetyScore(BaseModel):
    binary_score: str = Field(description="Workout is safe? 'yes' or 'no'")
    reason: str = Field(description="Detailed explanation of why it is unsafe, or 'Passed' if safe.")

def safety_grader_node(state):
    print("---NODE: GRADING SAFETY---")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0,
        google_api_key=GRADER_API_KEY
    )
    structured_llm = llm.with_structured_output(SafetyScore)

# Prompt to check for contraindications
    system = """You are a Medical Safety Auditor.
    Compare the Proposed Workout against the User's Conditions and Research Context.
    Look for 'Forbidden Movements' or intensity levels that are dangerous for the condition.
    If the workout includes even ONE forbidden movement, score it as 'no'."""
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Conditions: {conditions}\nResearch: {research}\nWorkout: {generation}")
    ])

    grader_chain = grader_prompt | structured_llm


    result = grader_chain.invoke({
        "conditions": state["user_profile"]["conditions"],
        "research": state["research_context"],
        "generation": state["generation"]
    })

    # Return the results to the state
    return {
        "is_safe": result.binary_score, 
        "explanation": result.reason
    }
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import time

GRADER_API_KEY = os.getenv("GRADER_API_KEY")

class SafetyScore(BaseModel):
    binary_score: str = Field(description="Workout is safe? 'yes' or 'no'")
    reason: str = Field(description="Detailed explanation of why it is unsafe, or 'Passed' if safe.")

def safety_grader_node(state):
    time.sleep(2) #for the free tier
    print("---NODE: GRADING SAFETY---")

    generation = state.get("generation", "")
    safe_exercises = state.get("safe_exercises", {})
    conditions = state.get("user_profile", {}).get("conditions", [])
    truncated_research = state.get("research_context", "")[:2000]

    flattened_list = []
    if isinstance(safe_exercises, dict):
        for exercise_list in safe_exercises.values():
            flattened_list.extend(exercise_list)
    else:
        flattened_list = safe_exercises # Fallback

    # If the menu only has names, the grader will rely more on its own knowledge + research.
    safety_lookup = "\n".join([
        f"{ex.get('name')}: not suitable for {ex.get('not_suitable_for', [])}" 
        for ex in flattened_list if isinstance(ex, dict)
    ])

    # safety_lookup = "\n".join([
    #     f"{ex['name']}: not suitable for {ex.get('not_suitable_for', [])}" 
    #     for ex in safe_exercises
    # ])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0,
        google_api_key=GRADER_API_KEY
    )
    structured_llm = llm.with_structured_output(SafetyScore)

# Prompt to check for contraindications
    system = """"SYSTEM: You are a Medical Safety Auditor.
        YOUR TOOLS:

        PATIENT CONDITIONS: {conditions}

        RESEARCH GUIDELINES: {research}

        EXERCISE SAFETY TAGS: {safety_database}

        YOUR TASK: Check if any exercise in the generated plan {generation} is listed in the EXERCISE SAFETY TAGS as 'not suitable' for the PATIENT CONDITIONS.
        Also, check if the plan violates any RESEARCH GUIDELINES.

        RESULT: If an exercise is found to be unsuitable, score it as 'no' and explain exactly which exercise violated which rule."""
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Conditions: {conditions}\nResearch: {research}\nSafety Database: {safety_database}\nWorkout to Audit: {generation}")
    ])

    grader_chain = grader_prompt | structured_llm



    result = grader_chain.invoke({
        "conditions": conditions,
        "research": truncated_research,
        "safety_database": safety_lookup,
        "generation": generation
    })

    # Return the results to the state
    return {
        "is_safe": result.binary_score, 
        "explanation": result.reason
    }
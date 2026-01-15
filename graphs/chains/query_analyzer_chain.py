import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")

class UserProfile(BaseModel):
    """Extracted fitness profile from user input."""
    name: str = Field(default="User", description="The user's name if provided.")

    intent: str = Field(
        default="workout_request",
        description="One of: 'workout_request', 'greeting', or 'general_chat'")

    conditions: List[str] = Field(
        description="List of IDs from: [lowerback_pain, neck_pain, shoulder_pain, hypertension, type_2_diabetes, obesity, beginner, knee_pain]"
    )
    goals: List[str] = Field(
        default_factory=list,
        description="List of specific muscles or groups (e.g., ['glutes', 'core', 'back','lower body'])"
    )
    workout_type: str = Field(
        default="balanced",
        description="The style of workout: 'strength', 'mobility', 'cardio', or 'full_body'"
    )
    experience_level: str = Field(default="beginner", description="beginner, intermediate, or advanced.")
    is_medical_emergency: bool = Field(description="True if the user mentions red flags like chest pain or numbness.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
    )

#Pydantic class 
query_analyzer_chain = llm.with_structured_output(UserProfile)

#Prompt Template
QUERY_ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Medical Fitness Intake Specialist. 
    Analyze the user's request and extract their profile accurately.
     
    INTENT CATEGORIZATION:
    - If the user says hello, hi, or introduces themselves: intent = 'greeting'
    - If the user asks for a workout, exercise, or routine: intent = 'workout_request'
    - If the user asks a general question or follows up: intent = 'general_chat'
     
     GOAL EXTRACTION (Populate the 'goals' list):
    - If specific muscles are mentioned (e.g. "biceps"), add them to the list.
    - 'legs' or 'lower body' -> ['quads', 'hamstrings', 'glutes', 'calves']
    - 'upper body' -> ['chest', 'back', 'shoulders', 'arms']
    - 'core' or 'abs' -> ['abs']
    - If no muscles are mentioned or it's a greeting, return an empty list [].

    WORKOUT TYPE:
    - Set 'workout_type' to 'full_body' if they want to train everything.
    - Set 'workout_type' to 'strength', 'mobility', or 'cardio' based on their tone. Default to 'strength'.
    
    CRITICAL MAPPING RULES:
    - 'lower back', 'disc', 'spine' -> lowerback_pain
    - 'neck', 'cervical' -> neck_pain
    - 'shoulder', 'rotator cuff' -> shoulder_pain
    - 'blood pressure', 'BP' -> hypertension
    - 'sugar', 'diabetes' -> type_2_diabetes
    - 'weight loss', 'heavy', 'overweight' -> obesity
    - 'knee', 'acl' -> knee_pain
    - 'new', 'start', 'beginner', 'newbie' -> beginner
     
    If the input is just a greeting, leave conditions as an empty list and goal_muscle as 'none'.
    If the user mentions 'chest pain', 'numbness', or 'vision loss', set is_medical_emergency to True.
    """),
    ("human", "{question}")
])

query_analyzer = QUERY_ANALYZER_PROMPT | query_analyzer_chain
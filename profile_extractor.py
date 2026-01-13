import json
from typing import List, Optional
from pydantic import BaseModel, Field

# 1. DEFINE THE DATA STRUCTURE (The "Bucket List")
class UserProfile(BaseModel):
    name: Optional[str] = "User"
    detected_conditions: List[str] = Field(
        description="Must be IDs from our JSON: [lumbar_disc_herniation, hypertension, diabetes_type2, obesity, beginner, etc.]"
    )
    fitness_goal: str = Field(description="e.g., 'Glute focus', 'Weight loss', 'Mobility'")
    experience_level: str = Field(default="beginner", description="beginner, intermediate, or advanced")
    specific_pain_points: List[str] = Field(description="Body parts currently hurting")

# 2. THE EXTRACTION LOGIC
def extract_profile_from_input(user_text: str, model_client):
    """
    Takes raw user input and returns a structured UserProfile object.
    """
    
    system_prompt = """
    You are a Medical Fitness Intake Specialist. 
    Analyze the user's message and extract their fitness profile.
    
    MAP MESSY TERMS TO THESE IDs:
    - 'back pain' or 'disc' -> lumbar_disc_herniation
    - 'high blood pressure' -> hypertension
    - 'new to the gym' -> beginner
    - 'neck' -> neck_pain
    
    Return the data in a valid JSON format.
    """

    # This is a conceptual call - you would use your Gemini/OpenAI client here
    response = model_client.chat.completions.create(
        model="gemini-1.5-flash",
        response_model=UserProfile, # This ensures the output matches our class!
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )
    
    return response
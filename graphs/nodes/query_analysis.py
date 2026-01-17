from graphs.chains.query_analyzer_chain import query_analyzer

def query_analyzer_node(state):
    """
    Node to analyze the user query and update the state with a structured profile.
    """
    print("---NODE: ANALYZING USER QUERY---")

    messages = state.get("messages", [])
    
    if not messages:
        # Fallback for the very first run if 'messages' isn't populated
        # This usually only happens if the input dict is wrong
        last_user_message = state.get("question", "No question provided")
    else:
        last_user_message = messages[-1].content


    #Pydantic object will be returned (UserProfile)
    extracted_profile = query_analyzer.invoke({"chat_history": messages[:-1],"question": last_user_message})

    profile_dict = {
        "name": extracted_profile.name,
        "intent": extracted_profile.intent,
        "conditions": extracted_profile.conditions,
        "goals": extracted_profile.goals,
        "workout_type": extracted_profile.workout_type,
        "level": extracted_profile.experience_level,
        "emergency": extracted_profile.is_medical_emergency
    }

    print(f"DEBUG: Query Analyzer produced dict: {profile_dict}") #debug line

    #if we have an existing user profile
    existing_profile = state.get("user_profile")

    #whether the local data is enough or web_search or different intent with query
    if extracted_profile.intent in ["greeting", "general_chat"]:
        datasource = "none" # Skip retrieval entirely
    elif not extracted_profile.conditions and "workout" not in last_user_message.lower() and not existing_profile:
        datasource = "web_search"
    else:
        datasource = "local_db"

    if "workout" in last_user_message.lower() or extracted_profile.goals:
        datasource = "local_db"

    #update the graph state
    return {
        "user_profile": profile_dict,
        "datasource": datasource
    }




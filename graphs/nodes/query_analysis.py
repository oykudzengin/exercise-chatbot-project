from graphs.chains.query_analyzer_chain import query_analyzer

def query_analyzer_node(state):
    """
    Node to analyze the user query and update the state with a structured profile.
    """
    print("---NODE: ANALYZING USER QUERY---")
    question = state["question"]

    #Pydantic object will be returned (UserProfile)
    extracted_profile = query_analyzer.invoke({"question": question})

    profile_dict = {
        "name": extracted_profile.name,
        "conditions": extracted_profile.conditions,
        "goal": extracted_profile.goal_muscle,
        "level": extracted_profile.experience_level,
        "emergency": extracted_profile.is_medical_emergency
    }

    #whether the local data is enough or not for the query
    datasource = "local_db"
    if not extracted_profile.conditions and "workout" not in question.lower():
        datasource = "web_search"

    #update the graph state
    return {
        "user_profile": profile_dict,
        "datasource": datasource
    }




import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "fitness-chatbot"

def retriever_node(state):
    print("---NODE: RETRIEVING & FILTERING---")
    
    #Get input from State
    profile = state.get("user_profile", {})
    conditions = profile.get("conditions", [])
    goals = profile.get("goals", [])
    print(f"DEBUG: AI extracted goals: {goals}") #debugging print line


    json_path = os.path.join("data", "database", "exercises_f.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_exercises = json.load(f)

    #muscle must match and no unsuitable condition is allowed
    safe_list = []
    for ex in all_exercises:
        search_text = f"{ex.get('name', '')} {ex.get('primary_muscles', '')} {ex.get('secondary_muscles', '')} {ex.get('body_part', '')}".lower()

                # Check if ANY of our goals appear in that text
        is_match = any(goal.lower()[:4] in search_text for goal in goals)
        
        if is_match:
            is_unsafe = any(cond in ex.get("not_suitable_for", []) for cond in conditions)
            if not is_unsafe:
                safe_list.append(ex)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )

    query = f"Safety guidelines and exercise precautions for {', '.join(conditions)}"
    docs = vectorstore.similarity_search(query, k=3)
    research_text = "\n\n".join([doc.page_content for doc in docs])

    return {
        "safe_exercises": safe_list[:7],
        "research_context": research_text
    }


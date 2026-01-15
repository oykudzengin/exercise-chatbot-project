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


    json_path = os.path.join("data", "database", "exercises_f.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_exercises = json.load(f)

    #muscle must match and no unsuitable condition is allowed
    safe_list = []
    for ex in all_exercises:
        primary = ex.get("primary_muscles", "").lower()
        secondary = ex.get("secondary_muscles", "").lower()
        
        match_found = any(g.lower() in primary or g.lower() in secondary for g in goals)
        if match_found:
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
        "safe_exercises": safe_list[:12],
        "research_context": research_text
    }


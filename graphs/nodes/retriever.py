import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "fitness-chatbot"

def retriever_node(state):
    print("---NODE: RETRIEVING & FILTERING---")
    
    #Get input from State
    profile = state["user_profile"]
    conditions = profile.get("conditions", [])
    target_muscle = profile.get("goal", "").lower()

    json_path = os.path.join("data", "database", "exercises_f.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_exercises = json.load(f)

    #muscle must match and no unsuitable condition is allowed
    safe_list = [
        ex for ex in all_exercises 
        if target_muscle in [m.lower() for m in ex.get('primary_muscles', [])]
        and not any(risk in ex.get('not_suitable_for', []) for risk in conditions)
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )

    query = f"Safety guidelines and exercise precautions for {', '.join(conditions)}"
    docs = vectorstore.similarity_search(query, k=3)
    research_text = "\n\n".join([doc.page_content for doc in docs])

    return {
        "safe_exercises": safe_list[:5], 
        "research_context": research_text
    }


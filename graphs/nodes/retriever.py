import json
import os
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "fitness-chatbot"

LEVEL_EQUIPMENT_MAP = {
    "beginner": ["band", "body weight", "machine", "sled machine"],
    "intermediate": ["machine", "barbell", "dumbbell", "weighted", "smith machine", "cable", "kettlebell"],
    "advanced": ["machine", "barbell", "dumbbell", "weighted", "smith machine", "cable", "kettlebell", "rope"]
}

def retriever_node(state):
    #print("---NODE: RETRIEVING & FILTERING---")
    
    #Get input from State
    profile = state.get("user_profile", {})
    conditions = profile.get("conditions", [])
    user_level = profile.get("level", "beginner").lower()
    #goals = profile.get("goals", [])
    #print(f"DEBUG: AI extracted goals: {goals}") #debugging print line
    #workout_request = profile.get("workout_type", "strength").lower()
    user_goal = str(profile.get("goals", [])) + str(profile.get("workout_type", "strength"))
    user_goal = user_goal.lower()

    current_dir = os.path.dirname(__file__)
    json_path = os.path.normpath(os.path.join(current_dir, "../../data/database/exercises_s2.json"))
    with open(json_path, 'r', encoding='utf-8') as f:
        all_exercises = json.load(f)

    allowed_equip = LEVEL_EQUIPMENT_MAP.get(user_level, ["body weight"])

    safe_pool = []
    for ex in all_exercises:
        # Check medical safety
        is_unsafe = any(cond in ex.get("not_suitable_for", []) for cond in conditions)
        
        # Check level matching (Equipment check)
        is_right_level = ex.get("equipment") in allowed_equip
        
        # Advanced check: specifically check the 'difficulty' tag for Advanced users
        if user_level == "advanced" and ex.get("difficulty").lower() != "advanced":
            is_right_level = False

        if not is_unsafe and is_right_level:
            safe_pool.append(ex)
    
    menu = {}

    all_muscle_names = set()
    for ex in all_exercises:
        # Assuming primary_muscles might be a string "abs" or list ["abs"]
        m = ex.get("primary_muscles", "")
        if isinstance(m, list): all_muscle_names.update([i.lower() for i in m])
        else: all_muscle_names.add(m.lower())

    # 2. Check if user mentioned a specific muscle (e.g., "back", "abs", "glutes")
    matched_muscle = next((m for m in all_muscle_names if m in user_goal), None)

    if matched_muscle:
        # Filter for ONLY that muscle
        muscle_pool = [
            ex['name'] for ex in safe_pool 
            if matched_muscle in str(ex.get("primary_muscles", "")).lower()
        ]
        
        # Shuffle and pick up to 5 for a focused workout
        random.shuffle(muscle_pool)
        menu[matched_muscle.upper()] = muscle_pool[:5]

    # A. LOWER BODY logic
    elif "lower" in user_goal:
        patterns = ["squat", "hinge", "extension", "calves"]
        for p in patterns:
            matches = [ex['name'] for ex in safe_pool if ex.get('pattern') == p and ex.get('type') == 'strength']
            if matches:
                random.shuffle(matches)
                menu[p.upper()] = matches[:4]                


    # B. UPPER BODY logic
    elif "upper" in user_goal:
        target_parts = ["shoulders", "back", "arms", "chest", "core"]   
        for part in target_parts:
        # 1. Find all safe exercises for THIS specific part
            matches = [ex['name'] for ex in safe_pool if ex.get('body_part') == part]
        
            if matches:
                # 2. Randomize to provide variety
                random.shuffle(matches)
                # 3. Add to the menu so the LLM sees: "CHEST: [Push-up, Bench Press]"
                menu[part.upper()] = matches[:2]

    # C. CARDIO logic
    elif "cardio" in user_goal:
        cardio_pool = [ex['name'] for ex in safe_pool if ex.get('type') == 'cardio']
        # Pick 4 random exercises, or all if pool is smaller than 4
        num_to_pick = min(len(cardio_pool), 4)
        menu["CARDIO_OPTIONS"] = random.sample(cardio_pool, num_to_pick)

    # D. MOBILITY logic
    elif "mobility" in user_goal:
        mobility_pool = [ex['name'] for ex in safe_pool if ex.get('type') == 'mobility']
        num_to_pick = min(len(mobility_pool), 6)
        menu["MOBILITY_PROGRAM"] = random.sample(mobility_pool, num_to_pick)      

    # E. ADVANCED 
    elif "advanced" in user_goal or "pro" in user_goal:
        # If advanced, we just give a mix of everything
        menu["ADVANCED_POOL"] = [ex['name'] for ex in safe_pool if ex.get('difficulty').lower() == "advanced"][:10]

    # F. DEFAULT FULL-BODY
    else:
        target_parts = ["shoulders", "back", "arms", "chest", "core", "legs"]
        for part in target_parts:
            matches = [ex['name'] for ex in safe_pool if ex.get('body_part') == part]
            if matches:
                random.shuffle(matches)
                menu[part.upper()] = matches[:6]
        

    #print(f"DEBUG: Programmed Menu Categories: {list(menu.keys())}")  

    
    #muscle must match and no unsuitable condition is allowed
    # safe_list = []
    # for ex in all_exercises:
    #     search_text = f"{ex.get('name', '')} {ex.get('primary_muscles', '')} {ex.get('secondary_muscles', '')} {ex.get('body_part', '')}".lower()

    #             # Check if ANY of our goals appear in that text
    #     is_match = any(goal.lower()[:4] in search_text for goal in goals)
        
    #     if is_match:
    #         is_unsafe = any(cond in ex.get("not_suitable_for", []) for cond in conditions)
    #         if not is_unsafe:
    #             safe_list.append(ex)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )

    query = f"Safety guidelines and exercise precautions for {', '.join(conditions)}"
    docs = vectorstore.similarity_search(query, k=3)
    research_text = "\n\n".join([doc.page_content for doc in docs])

    return {
        "safe_exercises": menu,
        "research_context": research_text
    }


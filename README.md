# 🏋️‍♂️ Elite Medical Fitness Coach
### *A Multi-Agent RAG System for Safe, Research-Backed Physical Activity*

Elite Medical Fitness Coach is a sophisticated AI-driven platform that bridges the gap between clinical research and daily fitness programming. Built using a **Retrieval-Augmented Generation (RAG)** architecture, the system synthesizes academic research with a curated exercise database to provide safe, personalized workout plans—even for users with chronic pain or medical conditions.

![App Header or Screenshot](https://github.com/user-attachments/assets/13333ce2-fc7d-41f0-9d1a-68f1bede84ac)


## 🌟 Key Features
* **Medical Safety Auditing:** A dedicated "Safety Grader" node that cross-references all AI suggestions against clinical guidelines.
* **Adaptive Persona:** An empathetic, coaching-style interface that provides "Clinical Tips" for every exercise.
* **Live Patient Dashboard:** A real-time UI that displays the AI’s extraction of user goals, experience levels, and health conditions.
* **Dynamic Knowledge Retrieval:** Uses LangChain to query specialized medical PDFs for conditions like Hypertension, Diabetes, and Musculoskeletal pain.

---

## :syringe: Considered Health Conditions

Since this is a learning demo project, the pool of the health conditions considered in this project is limited by few: </br>
* Hypertension
* Type 2 Diabetes
* Obesity
* Lower Back Pain
* Knee Pain
* Shoulder Pain
* Neck Pain

---

## :books: Health Conditions and Personal Trainer Data

Summarized data from the research papers about the considered conditions (mentioned above) is pulled and distinct text files are created to be sent to VectorDB. <br>
For Personal trainer guide, two PDF books are added for LLM to provide data with a coaching tone and fulfilling the generated workout plan by giving related information from the books.

---

## :pencil: Exercise Database

Exercise list is pulled from Kaggle in the first place. Wide range of exercises got narrowed down into fundamental exercise patterns for different body parts and exercise types. </br>
To obtain clear generated answer from the model, the exercises list filtered to be basic level. Query is getting analyzed and the intent is getting classified due to user's aim. Then spesific workout plan structures are determined by the retriever. </br>
| Workout Type | Body Parts | Patterns |
| :--- | :--- | :--- |
| **Lower Body Workout** | Quads, Hamstrings, Calves, Glutes | Hinge, Squat, Extension, Calves |
| **Upper Body Workout** | Shoulders, Back, Chest, Arms, Abs | Push, Pull, Core |
| **Cardio Workout** | Cardiovascular System  | Cardio |
| **Pro Workout** | Full body advanced | Advanced exercises |
| **Default Full Body Workout** | Full Body | All patterns, beginner level |

---

## 🏗️ The Technology Stack
| Component | Technology |
| :--- | :--- |
| **Orchestration** | [LangGraph](https://www.langchain.com/langgraph) (Stateful Multi-Agent Workflows) |
| **LLM** | [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Vector Database** | [Pinecone](https://www.pinecone.io/) / Local JSON |
| **Embeddings** | [Hugging Face](https://huggingface.co/) |

---

## 🚦 System Architecture
The project utilizes a directed acyclic graph (DAG) to manage the logic flow:
1.  **Greeting Node:** Manages the initial user onboarding.
2.  **Query Analyzer:** Extracts structured medical and fitness data from natural language.
3.  **Local Retriever:** Filters a 500+ exercise database based on "Pattern" and "Safe-for" tags.
4.  **Generator:** Synthesizes the workout plan and research-based clinical tips.
5.  **Safety Grader:** Acts as a final gatekeeper to prevent contraindicated movements.



---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Google Gemini API Key
* Tavily API Key
* Pinecone API Key
* Streamlit

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/medical-fitness-coach.git](https://github.com/yourusername/medical-fitness-coach.git)
    cd medical-fitness-coach
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    GRADER_API_KEY=your_gemini_api_key_here
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app_ui.py
    ```

---

## 📂 Project Structure
```text
exercise_chatbot_project/
├── main.py              # Compiled LangGraph Workflow
├── app_ui.py            # Streamlit Frontend
├── ingestion.py         # Database Index
├── requirements.txt     # Dependencies
├── graphs/
│   ├── state.py         # Shared Graph State
|   ├── chains/
|   |   ├── query_analyzer_chain.py
│   ├── nodes/           # Individual Logic Nodes (Greeting, Analysis, etc.)
|   |   ├── generator.py
|   |   ├── greeting.py
|   |   ├── query_analysis.py
|   |   ├── retriever.py
|   |   ├── safetY_grader.py
|   |   ├── web_search.py
├── data/               # Exercise JSON and Knowledge Base
│   ├── database/        
│   |   ├── conditions.json
│   |   ├── exercises_s2.json
│   ├── knowledge_base/
│   |   ├── pdf1.json
│   |   ├── pdf2.json
│   |   ├── diabetes.json
│   |   ├── hypertension.json
│   |   ├── knee.json
│   |   ├── lower_back.json
│   |   ├── neck.json
│   |   ├── obesity.json
│   |   ├── shoulders.json
└── .env                 # API Credentials

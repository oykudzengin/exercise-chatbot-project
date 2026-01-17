# 🏋️‍♂️ Elite Medical Fitness Coach
### *A Multi-Agent RAG System for Safe, Research-Backed Physical Activity*

Elite Medical Fitness Coach is a sophisticated AI-driven platform that bridges the gap between clinical research and daily fitness programming. Built using a **Retrieval-Augmented Generation (RAG)** architecture, the system synthesizes academic research with a curated exercise database to provide safe, personalized workout plans—even for users with chronic pain or medical conditions.

![App Header or Screenshot](path/to/your/screenshot.png)

## 🌟 Key Features
* **Medical Safety Auditing:** A dedicated "Safety Grader" node that cross-references all AI suggestions against clinical guidelines.
* **Adaptive Persona:** An empathetic, coaching-style interface that provides "Clinical Tips" for every exercise.
* **Live Patient Dashboard:** A real-time UI that displays the AI’s extraction of user goals, experience levels, and health conditions.
* **Dynamic Knowledge Retrieval:** Uses LangChain to query specialized medical PDFs for conditions like Hypertension, Diabetes, and Musculoskeletal pain.

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
my_fitness_project/
├── main.py              # Compiled LangGraph Workflow
├── app_ui.py            # Streamlit Frontend
├── graphs/
│   ├── state.py         # Shared Graph State
│   └── nodes/           # Individual Logic Nodes (Greeting, Analysis, etc.)
├── data/
│   └── database/        # Exercise JSON and Knowledge Base
└── .env                 # API Credentials
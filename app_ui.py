import streamlit as st
from main import ex_chatbot_app 
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Elite Medical Coach", page_icon="🏋️‍♂️")

#Sidebar: Patient Dashboard
with st.sidebar:
    st.title("📋 Patient Dashboard")
    st.info("This panel shows the AI's real-time extraction of your profile.")
    
    if st.button("🔄 Reset Conversation"):
        # Clears the Streamlit state and triggers a rerun
        st.session_state.messages = []
        st.session_state.user_profile = {}
        # Changing thread_id forces LangGraph to forget the history
        st.session_state.thread_id = f"session_{int(st.time.time())}"
        st.rerun()

    st.divider()
    
    # Placeholder for Profile Data
    profile = st.session_state.get("user_profile", {})
    st.subheader("Current Profile")
    st.write(f"**Experience:** {profile.get('level', 'Unknown')}")
    st.write(f"**Goals:** {', '.join(profile.get('goals', ['Not set']))}")
    st.write(f"**Conditions:** {', '.join(profile.get('conditions', ['None reported']))}")
    
    st.divider()
    st.caption("Built with LangGraph & Gemini 2.5 Flash")


st.title("Elite Medical Fitness Coach")

# 1. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_default"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle First-Time Greeting (Opening Speech)
if not st.session_state.messages:
    with st.spinner("Connecting to Coach..."):
    # Trigger the graph with an empty input to get the greeting
        initial_input = {"messages": []}
        for output in ex_chatbot_app.stream(initial_input, config):
            for key, value in output.items():
                if "messages" in value:
                    greeting_text = value["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": greeting_text})
                    st.rerun()

# 4. Chat Input
if prompt := st.chat_input("Tell me about your workout goals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Run LangGraph
    with st.chat_message("assistant"):
        with st.status("Coach is thinking...", expanded=False) as status:
            inputs = {"messages": [HumanMessage(content=prompt)]}
            final_answer = ""
            
            # Stream the graph execution
            for output in ex_chatbot_app.stream(inputs, config):
                for node_name, state_update in output.items():
                    # UPDATE DASHBOARD: If the Analyzer node finished, update the sidebar data
                    if node_name == "analyze_query" and "user_profile" in state_update:
                        st.session_state.user_profile = state_update["user_profile"]
                        # We don't rerun here so the flow continues to the generator
            
                    # CAPTURE FINAL ANSWER
                    if "generation" in state_update:
                        final_answer = state_update["generation"]
                    elif not final_answer and "messages" in state_update:
                        final_answer = state_update["messages"][-1].content

            status.update(label="Workout Generated!", state="complete", expanded=False)

        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.rerun()
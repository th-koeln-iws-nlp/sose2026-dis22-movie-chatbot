import streamlit as st
from llm.dummy_llm import DummyLLM

st.set_page_config(page_title="Movie Chatbot", page_icon=":material/movie:")

WELCOME_MESSAGE = "Hi! I am your friendly movie chatbot. Ask me anything about movies!"

LLM_OPTIONS = ["Dummy LLM", "Gemini 2.5 Flash", "Mistral Small 3.2"]
LLM_OPTIONS_AVAILABLE = {"Dummy LLM"}


def init_messages():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    selected_llm = st.selectbox(
        "Model",
        options=LLM_OPTIONS,
        format_func=lambda x: x if x in LLM_OPTIONS_AVAILABLE else f"{x} (coming soon)",
    )

    if selected_llm not in LLM_OPTIONS_AVAILABLE:
        st.info(f"{selected_llm} is not available yet. Using Dummy LLM.")
        selected_llm = "Dummy LLM"

    if st.button("Clear chat"):
        init_messages()

# Reinitialize messages when the model changes
if st.session_state.get("selected_llm") != selected_llm:
    st.session_state.selected_llm = selected_llm
    init_messages()

# Initialize on first run
if "messages" not in st.session_state:
    init_messages()

# Instantiate the active LLM
llm = DummyLLM()

# --- Main area ---
st.title("Movie Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask me about movies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = llm.generate_response(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

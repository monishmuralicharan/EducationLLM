import streamlit as st
from testLLM import RAGPipeline

rag_pipeline = RAGPipeline()

st.title("LLM app")

if 'conversation' not in st.session_state:
    st.session_state.conversation = ""

if "conversation_num" not in st.session_state:
    st.session_state.conversation_num = 1

prompt = st.chat_input("Enter Query")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.spinner("Loading..."):
        response = rag_pipeline.run_pipeline(prompt, st.session_state.conversation)
        st.session_state.conversation += "prompt " + str(st.session_state.conversation_num) + ": " + prompt + "\n" + "response " + str(st.session_state.conversation_num) + ": " + response + "\n"
        st.session_state.conversation_num += 1
    with st.chat_message("assistant"):
        st.write(response)
    
import streamlit as st
from testLLM import RAGPipeline

rag_pipeline = RAGPipeline()

st.title("LLM app")

state_context = ""

user_query = st.text_input("Enter your query:")

if user_query:
    with st.spinner("retrieving information..."):
        response = rag_pipeline.run_pipeline(user_query)
    st.write(response)


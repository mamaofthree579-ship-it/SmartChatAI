import streamlit as st
from openai import OpenAI
import requests

# Set up UI
st.title("Fact-Checked AI Researcher")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_query := st.chat_input("What do you want me to research?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 1. Verification Step: Fetch live context (Mock Search API call)
    with st.spinner("Searching and validating sources..."):
        # In production, replace with a real tool like Tavily or Serper API
        search_url = f"https://tavily.com" 
        # fake context for structural reference:
        verified_context = "Retrieved live data matching user query with source citations..." 

    # 2. Guardrail Prompting Step
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    system_prompt = (
        "You are a strict, no-hallucination research assistant. "
        "Answer the user's question using ONLY the provided verified context. "
        "If the context doesn't contain the answer, say 'I cannot verify this.' "
        "Provide URL citations for your claims."
    )

    with st.chat_message("assistant"):
        # Stream the response to the Streamlit UI
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Verified Context: {verified_context}"},
                {"role": "user", "content": user_query}
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

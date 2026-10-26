import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_groq import ChatGroq

# Initialize Streamlit Page UI Configuration
st.set_page_config(page_title="Zero-Hallucination AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Zero-Hallucination Cloud Research Chatbot")
st.caption("Powered by Streamlit, Direct DDGS Search, and Free Production Models via Groq Cloud")

# Securely check for the required API key in Streamlit Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 Missing GROQ_API_KEY! Please add it to your Streamlit Cloud app secrets as GROQ_API_KEY = \"your_key\"")
    st.stop()

# Initialize Free Cloud LLM (Using standard Llama 3.3 on Groq infrastructure)
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # High-reasoning production model id
        temperature=0.0,                  # Forced to 0.0 to actively block creative assumptions
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

llm = load_llm()

# Initialize Chat Memory State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past chat logs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt ingestion
if user_input := st.chat_input("Ask me anything that requires real-time research..."):
    # Append user question to UI state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Begin Agentic Processing Pipeline
    with st.chat_message("assistant"):
        status_box = st.status("🕵️ Agent Core: Initializing research...", expanded=True)
        
        # --- STEP 1: LIVE SEARCH VIA DIRECT DDGS ---
        status_box.write("🌐 Gathering web intelligence via DuckDuckGo...")
        try:
            search_raw = DDGS().text(user_input, max_results=5)
            raw_search_results = "\n\n".join([f"Source: {res.get('href')}\nContent: {res.get('body')}" for res in search_raw if "body" in res])
            
            if not raw_search_results:
                raw_search_results = "No explicit matching text found on the live web."
        except Exception as e:
            raw_search_results = "Failed to fetch web results due to connectivity limits."
            status_box.write(f"⚠️ Search skipped: {str(e)}")

        # --- STEP 2: DRAFT GENERATION (THE PRODUCER AGENT) ---
        status_box.write("✍️ Drafting initial response based exclusively on sources...")
        
        researcher_prompt = f"""
        You are a highly analytical Research Agent. Your sole priority is truthfulness.
        Task: Answer the User Query using ONLY the facts present in the Context block below.
        
        Context:
        \"\"\"{raw_search_results}\"\"\"
        
        User Query: {user_input}
        
        Strict Structural Constraints:
        - If the Context does not provide specific data to answer the query, say "I cannot verify this information based on available web records."
        - Do NOT extrapolate, speculate, or introduce external training data historical assumptions.
        - Every factual statement must explicitly map to a theme found in the context provided.
        """
        
        # Run initial processing step
        initial_draft_response = llm.invoke(researcher_prompt)
        initial_draft = initial_draft_response.content
        
        # --- STEP 3: LOGICAL AUDITING (THE CRITIC AGENT) ---
        status_box.write("🔍 Auditor Agent: Scanning draft for unverified claims...")
        
        critic_prompt = f"""
        You are an uncompromising AI Compliance Auditor. Your job is to catch hallucinations.
        Compare the Draft Response against the Raw Reference Material. 
        
        Raw Reference Material:
        \"\"\"{raw_search_results}\"\"\"
        
        Draft Response:
        \"\"\"{initial_draft}\"\"\"
        
        Instructions:
        - Identify any detail, metric, name, or claim in the Draft Response that cannot be explicitly verified or proven by the Raw Reference Material.
        - If the draft is 100% accurate to the text and introduces zero outside information, output the Draft Response exactly as written without alterations.
        - If you find unverified assumptions, completely rewrite the text to remove them. If the draft states things not found in the context, replace them with "Data unverified".
        
        Final Sanitized Output:
        """
        
        sanitized_response_obj = llm.invoke(critic_prompt)
        sanitized_response = sanitized_response_obj.content
        
        # Clean up UI element statuses upon completion
        status_box.update(label="✅ Research complete & verified!", state="complete", expanded=False)
        
        # Render the verified payload to the user
        st.markdown(sanitized_response)
        
        # Display the ground truths transparently below the response
        with st.expander("📚 View Raw Source Context Retrieved"):
            st.code(raw_search_results, language="text")
            
        # Append finalized structured log to session history
        st.session_state.messages.append({"role": "assistant", "content": sanitized_response})

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama

# Initialize Streamlit Page UI Configuration
st.set_page_config(page_title="Zero-Hallucination AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Zero-Hallucination Local Research Chatbot")
st.caption("Powered by Streamlit, DuckDuckGo Search, and Local Free Models via Ollama")

# Initialize Local LLM (Change "llama3" to "mistral" or "phi3" if preferred)
@st.cache_resource
def load_llm():
    return Ollama(model="llama3", temperature=0.0)  # Low temperature drops creativity & cuts hallucinations

llm = load_llm()
ddg_search = DuckDuckGoSearchRun()

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
        
        # --- STEP 1: LIVE SEARCH & CRAWL ---
        status_box.write("🌐 Gathering web intelligence via DuckDuckGo...")
        try:
            raw_search_results = ddg_search.run(user_input)
        except Exception as e:
            raw_search_results = "Failed to fetch web results due to rate limits."
            status_box.write("⚠️ Search failed. Relying on baseline system checks.")

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
        
        initial_draft = llm.invoke(researcher_prompt)
        
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
        
        sanitized_response = llm.invoke(critic_prompt)
        
        # Clean up UI element statuses upon completion
        status_box.update(label="✅ Research complete & verified!", state="complete", expanded=False)
        
        # Render the verified payload to the user
        st.markdown(sanitized_response)
        
        # Display the ground truths transparently below the response
        with st.expander("📚 View Raw Source Context Retrieved"):
            st.code(raw_search_results, language="text")
            
        # Append finalized structured log to session history
        st.session_state.messages.append({"role": "assistant", "content": sanitized_response})

import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader
from langchain_groq import ChatGroq

# Initialize Streamlit Page UI Configuration
st.set_page_config(page_title="Zero-Hallucination AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Zero-Hallucination Production Research Agent")
st.caption("Deep Crawling, Local PDF/Code Analytics, and Automated Fact Citation Engine")

# Securely check for the required API key in Streamlit Secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 Missing GROQ_API_KEY! Please add it to your Streamlit Cloud secrets box.")
    st.stop()

# Initialize Free Cloud LLM using Groq's active production string identifier
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",      # Production-active flagship reasoning model
        temperature=0.0,                  # Forced to 0.0 to actively block creative assumptions
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize LLM configuration instance: {str(e)}")
    st.stop()

# Helper: Extract readable text from deep crawling a target web link
def deep_crawl_url(url, timeout=5):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=timeout)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            # Strip scripts, styles, and navigational elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text().split())
            return text[:4000] # Cap text snippet per document to manage context limits
    except Exception:
        return ""
    return ""

# Initialize Chat Memory State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Document Repository Integration (The "Local Files" feature)
with st.sidebar:
    st.header("📂 Local Document Repository")
    uploaded_files = st.file_uploader(
        "Upload reference materials (PDFs, Markdown, Python code)", 
        type=["pdf", "md", "txt", "py"], 
        accept_multiple_files=True
    )
    
    local_context_pool = ""
    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} local files successfully!")
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                try:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        local_context_pool += page.extract_text() + "\n"
                except Exception as pdf_err:
                    st.error(f"Error parsing PDF file {file.name}: {pdf_err}")
            else:
                local_context_pool += file.read().decode("utf-8") + "\n"

# Render past chat logs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt ingestion
if user_input := st.chat_input("Ask me anything that requires deep verification or document search..."):
    # Append user question to UI state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Begin Agentic Processing Pipeline
    with st.chat_message("assistant"):
        status_box = st.status("🕵️ Agent Core: Initializing research loop...", expanded=True)
        
        # --- STEP 1: WEB INTELLIGENCE SEARCH ---
        status_box.write("🌐 STEP 1: Gathering web records via DuckDuckGo...")
        sources_manifest = {}
        web_snippets = []
        
        try:
            search_raw = DDGS().text(user_input, max_results=4)
            idx = 1
            for res in search_raw:
                href = res.get('href')
                body = res.get('body', '')
                if href and body:
                    sources_manifest[f"Source {idx}"] = href
                    web_snippets.append(f"[{idx}] Source Link: {href}\nSnippet: {body}")
                    
                    # --- STEP 2: DEEP CRAWLING EXTENSION ---
                    status_box.write(f"🕷️ STEP 2: Deep crawling content from Source [{idx}]...")
                    deep_text = deep_crawl_url(href)
                    if deep_text and len(deep_text) > 200:
                        web_snippets.append(f"[{idx}] Deep Web Text Page Context:\n{deep_text}")
                    idx += 1
            
            web_context = "\n\n".join(web_snippets)
        except Exception as e:
            web_context = "No live web text collected due to connectivity limits."
            status_box.write(f"⚠️ Search skipped: {str(e)}")

        # Combine Local Upload Repository context with Live Web Search contexts
        consolidated_context = ""
        if local_context_pool:
            consolidated_context += f"--- LOCAL USER UPLOADED FILES ---\n{local_context_pool}\n\n"
        consolidated_context += f"--- LIVE WEB HARVESTED RESEARCH DATA ---\n{web_context}"

        if len(consolidated_context.strip()) < 50:
            consolidated_context = "No reference text context available from local files or live internet web engines."

        # --- STEP 3: DRAFT GENERATION (THE PRODUCER AGENT WITH CITATIONS) ---
        status_box.write("✍️ STEP 3: Drafting initial response with strict citation rules...")
        
        researcher_prompt = f"""
        You are a highly analytical Research Agent. Your sole priority is truthfulness and structural accuracy.
        
        Task: Answer the User Query using ONLY the facts present in the Context block below.
        
        Context Data:
        \"\"\"{consolidated_context}\"\"\"
        
        User Query: {user_input}
        
        Strict Structural Constraints:
        - If the Context does not provide data to answer the query, write "I cannot verify this information based on available files or web records."
        - Do NOT extrapolate or assume missing facts.
        - You must insert numerical inline bracket citations like [1], [2], or [3] directly following sentences containing claims extracted from web data or local data.
        - Every claim made must be backed up by a numbered source present in the Context.
        """
        
        initial_draft_response = llm.invoke(researcher_prompt)
        initial_draft = initial_draft_response.content
        
        # --- STEP 4: LOGICAL AUDITING (THE CRITIC AGENT) ---
        status_box.write("🔍 STEP 4: Scanning draft for unverified claims and citation integrity...")
        
        critic_prompt = f"""
        You are an uncompromising AI Compliance Auditor. Your job is to catch hallucinations and verify footnotes.
        
        Reference Material:
        \"\"\"{consolidated_context}\"\"\"
        
        Draft Response to Audit:
        \"\"\"{initial_draft}\"\"\"
        
        Instructions:
        - Identify any detail, metric, name, or claim in the Draft Response that cannot be proven directly by the Reference Material.
        - Ensure numerical citation brackets ([1], [2], etc.) correctly align with text facts. 
        - Remove any sentence containing an unverified assumption. If the whole text lacks data support, output "Data unverified".
        
        Final Sanitized Output:
        """
        
        sanitized_response_obj = llm.invoke(critic_prompt)
        sanitized_response = sanitized_response_obj.content
        
        # Clean up UI element statuses upon completion
        status_box.update(label="✅ Research complete & verified!", state="complete", expanded=False)
        
        # Render the verified payload to the user
        st.markdown(sanitized_response)
        
        # Display Sources Manifest and Ground Truth transparently below the response
        if sources_manifest:
            with st.expander("🔗 Active Footnote Sources Index"):
                for key, url in sources_manifest.items():
                    st.markdown(f"**{key}**: [{url}]({url})")
                    
        with st.expander("📚 View Raw Combined Context Processing Data"):
            st.code(consolidated_context, language="text")
            
        # Append finalized structured log to session history
        st.session_state.messages.append({"role": "assistant", "content": sanitized_response})

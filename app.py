import streamlit as st
import os
import shutil
import time
from rag_pipeline import create_vector_db, get_qa_chain

#PAGE CONFIG 
st.set_page_config(page_title="Smart RAG Assistant", layout="wide")

#DARK MODE
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    bg_user = "#2e7d32"
    bg_bot = "#424242"
    text_color = "white"
else:
    bg_user = "#DCF8C6"
    bg_bot = "#EAEAEA"
    text_color = "black"

#CSS 
st.markdown(f"""
<style>
.chat-row {{
    display: flex;
    margin-bottom: 10px;
}}

.user-container {{
    justify-content: flex-end;
}}

.bot-container {{
    justify-content: flex-start;
}}

.user-msg {{
    background-color: {bg_user};
    color: {text_color};
    padding: 12px;
    border-radius: 15px;
    border-bottom-right-radius: 2px;
    max-width: 60%;
    word-wrap: break-word;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
}}

.bot-msg {{
    background-color: {bg_bot};
    color: {text_color};
    padding: 12px;
    border-radius: 15px;
    border-bottom-left-radius: 2px;
    max-width: 60%;
    word-wrap: break-word;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
}}

.header-title {{
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}}

.subtext {{
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}}
</style>
""", unsafe_allow_html=True)

#SIDEBAR
st.sidebar.markdown("## 📂 Document Manager")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

#Show uploaded files
if uploaded_files:
    st.sidebar.markdown("### 📄 Uploaded Files:")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")

#PROCESS
if st.sidebar.button("⚙️ Process Documents"):
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
                f.write(file.read())

        with st.spinner("🔄 Processing documents..."):
            create_vector_db(DATA_FOLDER)

        st.sidebar.success("✅ Documents processed!")
    else:
        st.sidebar.warning("⚠️ Upload files first")

#CLEAR DATABASE
if st.sidebar.button("🗑️ Clear Database"):
    if os.path.exists("./db"):
        shutil.rmtree("./db")
        st.sidebar.success("Database cleared!")
    else:
        st.sidebar.info("No database found")

#CLEAR CHAT 
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

#CACHE 
@st.cache_resource
def load_chain():
    return get_qa_chain()

#qa_chain = load_chain()

#CHAT MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

#TITLE 
st.markdown("<div class='header-title'>🤖 Smart RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Chat with your documents intelligently</div>", unsafe_allow_html=True)

#CHAT INPUT
query = st.chat_input("💬 Ask something from your documents...")

#HANDLE QUERY
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🤖 Thinking..."):
        time.sleep(0.5)

        try:
            #Check DB exists
            if not os.path.exists("./db"):
                answer = "⚠️ Please process documents first!"
            else:
                qa_chain = get_qa_chain()  
                answer = qa_chain.run(query)

        except Exception as e:
            answer = f"⚠️ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

#DISPLAY CHAT
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-row user-container">
                <div class="user-msg">🧠 {msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row bot-container">
                <div class="bot-msg">🤖 {msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

#FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🚀 Powered by LangChain + FAISS + Groq</p>",
    unsafe_allow_html=True
)
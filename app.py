import streamlit as st
import os
import shutil
from rag_pipeline import create_vector_db, get_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical Chatbot", layout="wide")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "last_real_query" not in st.session_state:
    st.session_state.last_real_query = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Medical Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

if st.sidebar.button("⚙️ Process"):
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
                f.write(file.read())

        with st.spinner("Processing..."):
            create_vector_db(DATA_FOLDER)

        st.sidebar.success("✅ Done")
    else:
        st.sidebar.warning("Upload file first")

if st.sidebar.button("🗑️ Clear DB"):
    if os.path.exists("./db"):
        shutil.rmtree("./db")
        st.sidebar.success("Database Cleared!")

# ---------------- TITLE ----------------
st.title("🏥 AI Medical Chatbot")

# ---------------- INPUT ----------------
query = st.chat_input("Ask your question...")

if query:
    clean_query = query.strip().lower()

    # ✅ HANDLE YES
    if clean_query.startswith("yes"):

        if st.session_state.last_real_query == "":
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ No previous question found."
            })
        else:
            llm = ChatGroq(
                temperature=0,
                model="llama-3.1-8b-instant",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            detailed_prompt = f"""
You are a medical assistant.

User already asked:
{st.session_state.last_real_query}

Previous answer:
{st.session_state.last_answer}

Now give a MORE DETAILED explanation.

Include:
- Explanation
- Causes
- Symptoms
- Treatment

Keep it simple and human-friendly.
"""

            response = llm.invoke(detailed_prompt)

            answer = response.content if hasattr(response, "content") else str(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.last_answer = answer

    # ✅ HANDLE NO
    elif clean_query == "no":
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👍 Okay!"
        })

    # ✅ NORMAL QUESTION
    else:
        clean_query = query

        st.session_state.messages.append({
            "role": "user",
            "content": clean_query
        })

        try:
            if os.path.exists("./db"):
                qa = get_qa_chain()
                answer = qa.answer(clean_query)

                if answer == "NOT_FOUND":
                    st.session_state.pending_query = clean_query
                    answer = "⚠️ Not found in PDF. Do you want me to fetch from external knowledge?"
                else:
                    answer += "\n\n👉 If you want more details, just say YES."

            else:
                answer = "⚠️ Please process PDF first"

        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.last_real_query = clean_query
        st.session_state.last_answer = answer
# ---------------- DISPLAY ----------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


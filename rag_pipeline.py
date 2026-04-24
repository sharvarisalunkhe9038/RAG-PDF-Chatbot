import os
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# LOAD DOCUMENTS
def _collect_docs(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(fpath).load())

        elif fname.lower().endswith(".docx"):
            docs.extend(Docx2txtLoader(fpath).load())

    if not docs:
        raise ValueError("No documents found")

    return docs


# CREATE VECTOR DB
def create_vector_db(folder_path: str):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import FAISS

    # ✅ DEFINE FIRST
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ THEN CHECK DB
    if os.path.exists("./db"):
        return FAISS.load_local(
            "./db",
            embeddings,
            allow_dangerous_deserialization=True
        )

    # LOAD DOCS
    docs = _collect_docs(folder_path)

    # SPLIT (FAST)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # CREATE DB
    db = FAISS.from_documents(chunks, embedding=embeddings)
    db.save_local("./db")

    return db


# QA CLASS
class MedicalQA:
    def __init__(self, retriever, llm, k: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def answer(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        print("Retrieved docs:", len(docs))

        if not docs or len(docs) == 0:
            return "NOT_FOUND"
    

        context = "\n\n".join(d.page_content[:1000] for d in docs[:self.k])

        prompt = f"""
You are a friendly medical assistant chatbot.

Understand the user's question and answer accordingly.

RULES:
- If user asks definition → give short definition
- If user asks symptoms → give only symptoms
- If user asks treatment → give only treatment
- If user asks diet → give simple food suggestions
- If user asks explanation → give simple explanation
- DO NOT force all sections (Definition, Causes, etc.)
- Answer ONLY what is asked

LANGUAGE:
- Use very simple English (like explaining to a normal person)
- Avoid complex medical terms
- If needed, explain difficult words in simple way

FORMAT:
- Keep answer short and clear
- Use bullet points if needed
- Be human-friendly (like ChatGPT)

IMPORTANT:
- Use ONLY the provided context
- If no answer → return "NOT_FOUND"

Context:
{context}

Question: {query}

Answer:
"""

        response = self.llm.invoke(prompt)

        text = response.content if hasattr(response, "content") else str(response)

        if text.strip().upper() == "NOT_FOUND":
            return "NOT_FOUND"


        return text.strip()


def get_qa_chain():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "./db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant",
        groq_api_key=groq_key
    )

    return MedicalQA(retriever, llm)
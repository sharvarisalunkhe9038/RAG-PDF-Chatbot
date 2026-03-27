import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

#Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("❌ GROQ_API_KEY not found! Please add it to your .env file.")

#Load all PDF files
def _collect_docs(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(fpath).load())
        elif fname.lower().endswith(".docx"):
            docs.extend(Docx2txtLoader(fpath).load())
    if not docs:
        raise ValueError("⚠️ No PDF or DOCX files found in the folder!")
    return docs


#Create and store FAISS database
def create_vector_db(folder_path: str):
    import shutil

    #Clear old DB
    if os.path.exists("./db"):
        shutil.rmtree("./db")

    #Load documents
    docs = _collect_docs(folder_path)

    #Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    #Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    #Create FAISS DB
    db = FAISS.from_documents(chunks, embedding=embeddings)
    db.save_local("./db")

    return db


#Helper class for QA
class _SimpleQA:
    def __init__(self, retriever, llm, k: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def _answer(self, query: str) -> str:
        #Retrieve relevant chunks
        docs = self.retriever.invoke(query)
        print("DEBUG DOCS:", docs)
        
        if not docs:
            return "⚠️ No relevant context found in the documents."

        #Combine retrieved text
        context = "\n\n".join(
f"📄 {d.metadata.get('source','Document')}:\n{d.page_content}"
for d in docs[:self.k])

        #Create a clear prompt
        prompt = (
    "You are a helpful assistant.\n"
    "Answer based on the context. Try your best.\n"
    "If exact answer is not found, give the closest possible answer.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n"
    "Answer:"
)
        

        #Call Groq model
        response = self.llm.invoke(prompt)

        try:
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, dict) and "content" in response:
                return str(response["content"]).strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response)
        except Exception as e:
            return f"⚠️ Could not parse response properly: {e}"

    def run(self, query: str) -> str:
        return self._answer(query)

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        q = inputs.get("query", "")
        return {"result": self._answer(q)}


#Load the database and prepare QA chain
def get_qa_chain():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("./db", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant", 
        groq_api_key=groq_key
    )

    return _SimpleQA(retriever=retriever, llm=llm, k=3)

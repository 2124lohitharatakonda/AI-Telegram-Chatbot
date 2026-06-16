"""
LangChain RAG Chain — Retrieval-Augmented Generation for context-aware responses.
Combines FAISS retrieval with LLM generation for document Q&A.
"""

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "models/langchain_faiss"

SYSTEM_PROMPT = """You are NexBot, an intelligent and helpful AI assistant integrated with Telegram.
You answer questions clearly and concisely using the provided context.
If the context does not contain the answer, say so honestly — do not make up information.
Always maintain a friendly, professional tone.

Context from knowledge base:
{context}

Conversation history is available to maintain multi-turn awareness.
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\nUser Question: {question}\n\nNexBot Answer:",
)


# ---------------------------------------------------------------------------
# Document Loading & Splitting
# ---------------------------------------------------------------------------

DEMO_DOCS_TEXT = [
    ("faq.txt",
     "Our refund policy allows returns within 30 days. Damaged items must be reported within 48 hours. "
     "Refunds are processed in 5–7 business days. Free shipping on orders above ₹500."),
    ("x200_manual.txt",
     "Model X200 Setup Guide: 1) Connect power adapter. 2) Hold power button 3 seconds. "
     "3) Connect to WiFi using label password. Warranty: 2 years. Support: 1800-XXX-XXXX."),
    ("support.txt",
     "Support hours are Monday–Friday, 9 AM–6 PM IST. "
     "Email: support@nexbot.in. Live chat available on website. Avg response: 2 hours."),
]


def write_demo_docs():
    os.makedirs("documents", exist_ok=True)
    for name, content in DEMO_DOCS_TEXT:
        path = os.path.join("documents", name)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)


def load_and_split_documents(folder: str = "documents") -> list:
    write_demo_docs()
    loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(raw_docs)


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(docs: list = None):
    if docs is None:
        docs = load_and_split_documents()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = LangFAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved → {VECTOR_STORE_PATH} ({len(docs)} chunks)")
    return vector_store


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangFAISS.load_local(VECTOR_STORE_PATH, embeddings,
                                allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

def build_qa_chain(llm, vector_store) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def build_conversational_chain(llm, vector_store) -> ConversationalRetrievalChain:
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Response Formatting
# ---------------------------------------------------------------------------

def format_response(result: dict) -> dict:
    answer = result.get("result") or result.get("answer", "")
    sources = result.get("source_documents", [])

    return {
        "answer": answer.strip(),
        "sources": [
            {
                "file": os.path.basename(doc.metadata.get("source", "unknown")),
                "preview": doc.page_content[:100],
            }
            for doc in sources
        ],
    }


def retrieve_and_format(query: str, chain) -> str:
    result = chain({"query": query} if hasattr(chain, "run") else {"question": query})
    formatted = format_response(result)
    response = formatted["answer"]
    if formatted["sources"]:
        src_names = ", ".join(s["file"] for s in formatted["sources"])
        response += f"\n\n📄 Sources: {src_names}"
    return response


if __name__ == "__main__":
    print("LangChain RAG Chain module loaded.")
    print("To run: build_vector_store() → build_qa_chain(llm, vs) → retrieve_and_format(query, chain)")
    print("\nDemo document loading...")
    docs = load_and_split_documents()
    print(f"Loaded {len(docs)} document chunks")
    for d in docs[:2]:
        print(f"  [{d.metadata.get('source','?')}] {d.page_content[:80]}...")

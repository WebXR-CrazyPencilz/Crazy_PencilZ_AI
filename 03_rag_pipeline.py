import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("Step 1: Loading PDF...")
loader = PyPDFLoader("doc.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages from PDF")

print("\nStep 2: Chunking PDF...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

print("\nStep 3: Creating embeddings and storing in ChromaDB...")
print("(This may take 1-2 minutes...)")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_store"
)
print("Stored in ChromaDB successfully!")

print("\nStep 4: Setting up RAG chain...")
llm = ChatGroq(model="llama-3.3-70b-versatile")

system_prompt = """You are a helpful assistant.
Answer the question using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def ask_question(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return response.content

print("\nRAG Pipeline Ready!")
print("="*50)

while True:
    question = input("\nAsk a question about your PDF (or type 'exit'): ")
    if question.lower() == 'exit':
        break
    print("\nAnswer:")
    print(ask_question(question))
    print("="*50)
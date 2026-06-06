import os
import time
import base64
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datetime import datetime
import pytz

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from ddgs import DDGS

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
@app.route('/chatbot.html')
def serve_chatbot():
    return send_from_directory('.', 'chatbot.html')

# ── Uploads folder ────────────────────────────────────────
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder ready: {UPLOAD_FOLDER}")

print("Loading RAG pipeline...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("chroma_store"):
    print("Loading existing ChromaDB...")
    vector_store = Chroma(
        persist_directory="chroma_store",
        embedding_function=embeddings
    )
else:
    print("Creating new ChromaDB from PDF...")
    loader = PyPDFLoader("doc.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_store"
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.3-70b-versatile")
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

print("RAG Pipeline Ready!")

# ── Diet plans ────────────────────────────────────────────
diet_plans = {
    "high_cholesterol": "EAT: Oats, whole grains, fruits, vegetables, nuts, olive oil, fish\nAVOID: Fried foods, red meat, full-fat dairy, processed snacks, butter",
    "high_sugar":       "EAT: Leafy greens, beans, whole grains, lean protein, berries\nAVOID: Sugar, white rice, white bread, sweets, sugary drinks",
    "low_hemoglobin":   "EAT: Spinach, beetroot, pomegranate, dates, rajma, chicken, fish\nAVOID: Tea/coffee with meals, processed foods",
    "general":          "EAT: Balanced diet with fruits, vegetables, whole grains, lean protein\nAVOID: Processed foods, excess sugar, fried foods"
}

# ── Conversation memory ───────────────────────────────────
conversations = {}

def get_history(session_id):
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

def add_to_history(session_id, role, content):
    if session_id not in conversations:
        conversations[session_id] = []
    conversations[session_id].append({"role": role, "content": content})
    if len(conversations[session_id]) > 10:
        conversations[session_id] = conversations[session_id][-10:]

# ── Web search ────────────────────────────────────────────
def search_web(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=4)
            if results:
                return "\n\n".join([
                    f"Source: {r['href']}\n{r['title']}\n{r['body']}"
                    for r in results
                ])
    except Exception as e:
        print(f"Web search error: {e}")
    return None

def build_messages(session_id, context, question, source):
    if source == "pdf":
        system = f"""You are a warm, friendly and professional sales assistant for Brigade Stellaris,
a premium residential project by Brigade Group.

Your personality:
- Be enthusiastic about the project but honest
- Use simple, clear language — avoid jargon
- Keep responses concise but complete
- Remember previous questions and refer back when relevant
- End with a helpful follow-up suggestion when possible
- Use bullet points for lists

If the context doesn't contain enough information, respond with exactly: NOT_IN_PDF
Only respond NOT_IN_PDF if completely unrelated to Brigade Stellaris or real estate.
Note: You have access to real-time web search for non-property questions.

Context from document:
{context}"""
    else:
        system = f"""You are a warm, friendly and professional assistant with access to real-time web search.
You CAN access the internet and real-time information through web search.
Answer using the web search results provided below — these are LIVE results fetched right now.
Never say you cannot access the internet or real-time data — you can!
Remember the conversation history and refer back when relevant.
Use bullet points for lists. Be concise but helpful.

Live Web Search Results:
{context}"""

    messages = [{"role": "system", "content": system}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": question})
    return messages

# ── Routes ────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        time_keywords = ["what time", "current time", "what is the time", "ist time"]
        date_keywords = ["today's date", "current date", "what is today", "what date"]

        if any(kw in question.lower() for kw in time_keywords):
            answer = f"The current time in India (IST) is:\n\n🕐 {now.strftime('%I:%M:%S %p IST')}\n📅 {now.strftime('%A, %d %B %Y')}"
            add_to_history(session_id, "user", question)
            add_to_history(session_id, "assistant", answer)
            return jsonify({"answer": answer, "source": "web"})

        if any(kw in question.lower() for kw in date_keywords):
            answer = f"Today's date is:\n\n📅 {now.strftime('%A, %d %B %Y')}\n🕐 {now.strftime('%I:%M %p IST')}"
            add_to_history(session_id, "user", question)
            add_to_history(session_id, "assistant", answer)
            return jsonify({"answer": answer, "source": "web"})

        web_keywords = [
            "today", "this year", "today's", "this week", "this month",
            "next monday", "next tuesday", "next wednesday", "next thursday",
            "next friday", "next saturday", "next sunday", "yesterday",
            "tomorrow", "calendar", "schedule", "when is", "what day",
            "news", "weather", "stock", "price of", "latest", "who is",
            "when did", "what happened", "how much is", "current price",
            "can you access", "do you have access", "real time", "real-time",
            "search for", "look up", "find out", "check online",
            "google", "internet", "website", "online"
        ]
        force_web = any(kw in question.lower() for kw in web_keywords)

        if not force_web:
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            messages = build_messages(session_id, context, question, "pdf")
            response = llm.invoke(messages)
            answer = response.content.strip()
        else:
            answer = "NOT_IN_PDF"

        if "NOT_IN_PDF" in answer:
            print(f"Searching web for: {question}")
            web_results = search_web(question)
            if web_results:
                messages = build_messages(session_id, web_results, question, "web")
                web_response = llm.invoke(messages)
                answer = web_response.content
                add_to_history(session_id, "user", question)
                add_to_history(session_id, "assistant", answer)
                return jsonify({"answer": answer, "source": "web"})
            else:
                answer = "I couldn't find information about this. Could you rephrase your question?"
                return jsonify({"answer": answer, "source": "none"})

        add_to_history(session_id, "user", question)
        add_to_history(session_id, "assistant", answer)
        return jsonify({"answer": answer, "source": "pdf"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        session_id = request.form.get("session_id", "default")

        # ── Save image ────────────────────────────────────
        timestamp = int(time.time())
        safe_name = file.filename.replace(" ", "_")
        filename = f"{timestamp}_{safe_name}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.seek(0)
        file.save(save_path)
        print(f"✅ Image saved: {save_path}")

        # ── Encode for vision model ───────────────────────
        file.seek(0)
        image_data = base64.b64encode(file.read()).decode("utf-8")
        ext = os.path.splitext(file.filename)[1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        # ── Analyze with vision model ─────────────────────
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": "Analyze this image in detail. If it's a blood work report, extract all values and classify as HIGH/LOW/NORMAL. If it's food, describe its nutritional value. If it's a product, describe it. If it's something else, describe what you see clearly."
                }
            ]
        }

        print("Analyzing image with vision model...")
        vision_response = vision_llm.invoke([message])
        analysis = vision_response.content
        print("Vision analysis done!")

        # ── Get recommendation ────────────────────────────
        rec_prompt = f"Based on this image analysis:\n{analysis}\n\nProvide a friendly summary and helpful recommendation."
        rec_response = llm.invoke(rec_prompt)

        final_answer = (
            f"📸 Image Analysis:\n\n{analysis}\n\n"
            f"💡 Recommendation:\n\n{rec_response.content}\n\n"
            f"📁 Saved as: {filename}"
        )

        add_to_history(session_id, "user", f"[Uploaded image: {file.filename}]")
        add_to_history(session_id, "assistant", final_answer)

        return jsonify({"answer": final_answer, "source": "image"})

    except Exception as e:
        print(f"Image analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    data = request.json
    session_id = data.get("session_id", "default")
    if session_id in conversations:
        conversations[session_id] = []
    return jsonify({"status": "cleared"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
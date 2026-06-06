print("Script started...")

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API keys from .env file
load_dotenv()

print("Making Groq call...")

# ── 1. Simple call using Groq ────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile")

response = llm.invoke("How many moons does Jupiter have?")
print("\nGROQ RESPONSE:")
print(response.content)

# ── 2. Call with System Prompt ───────────────────────────────
messages = [
    ("system", "You are a helpful assistant that answers in one short line, less than 30 words."),
    ("human", "How many moons does Jupiter have?"),
]

response2 = llm.invoke(messages)
print("\nGROQ WITH SYSTEM PROMPT:")
print(response2.content)

# ── 3. Temperature demo ──────────────────────────────────────
llm_temp0 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_temp1 = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)

response3 = llm_temp0.invoke("Write a one line tagline for my coffee shop")
response4 = llm_temp1.invoke("Write a one line tagline for my coffee shop")

print("\nTEMPERATURE 0 (less creative):")
print(response3.content)

print("\nTEMPERATURE 1 (more creative):")
print(response4.content)
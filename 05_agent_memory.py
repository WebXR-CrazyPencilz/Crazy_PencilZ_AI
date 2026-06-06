from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Tools (same as before) ────────────────────────────────
products = {
    "laptop":     {"price": 999,  "brand": "Dell",    "ram": "16GB",    "storage": "512GB SSD"},
    "phone":      {"price": 699,  "brand": "Samsung", "battery": "5000mAh", "camera": "108MP"},
    "headphones": {"price": 199,  "brand": "Sony",    "type": "Wireless",   "battery": "30hrs"},
    "smartwatch": {"price": 299,  "brand": "Apple",   "features": "Health tracking, GPS"},
    "tablet":     {"price": 499,  "brand": "iPad",    "storage": "256GB",   "display": "10.9 inch"},
}

reviews = {
    "laptop":     {"rating": 4.5, "total": 1200, "summary": "Great performance, excellent build quality"},
    "phone":      {"rating": 4.3, "total": 890,  "summary": "Amazing camera, long battery life"},
    "headphones": {"rating": 4.7, "total": 2300, "summary": "Best sound quality, very comfortable"},
    "smartwatch": {"rating": 4.2, "total": 560,  "summary": "Good health tracking, stylish design"},
    "tablet":     {"rating": 4.6, "total": 780,  "summary": "Perfect for work and entertainment"},
}

@tool
def get_product(name: str) -> str:
    """Use this tool to get product information like price, brand,
    and specifications for a given product name."""
    name = name.lower().strip()
    if name in products:
        return str(products[name])
    return f"Product '{name}' not found. Available: {list(products.keys())}"

@tool
def get_reviews(name: str) -> str:
    """Use this tool to get customer reviews, ratings, and feedback
    for a given product name."""
    name = name.lower().strip()
    if name in reviews:
        return str(reviews[name])
    return f"No reviews found for '{name}'. Available: {list(reviews.keys())}"

# ── Create Agent WITH Memory ──────────────────────────────
tools = [get_product, get_reviews]
memory = MemorySaver()

agent_executor = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)

# ── Thread config = session ID ────────────────────────────
config = {"configurable": {"thread_id": "user_123"}}

def ask(question):
    print(f"\nYou: {question}")
    print("-"*40)
    result = agent_executor.invoke(
        {"messages": [("user", question)]},
        config=config
    )
    answer = result["messages"][-1].content
    print(f"Agent: {answer}")
    return answer

# ── Test memory with follow-up questions ─────────────────
print("\n" + "="*50)
print("AI AGENT WITH MEMORY")
print("="*50)

ask("What is the price of the laptop?")
ask("What are the reviews for it?")        # 'it' = laptop, tests memory!
ask("Which one has the highest rating?")   # should compare all products
ask("Should I buy it?")                    # 'it' = highest rated product
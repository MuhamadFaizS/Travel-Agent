import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# ==========================================
# 1. DATABASE IN-MEMORY
# ==========================================
customer_itinerary = []

# ==========================================
# 2. TOOLS
# ==========================================
@tool
def get_travel_catalog() -> str:
    """Provide the latest up-to-date travel catalog and services."""
    return """
    TRAVEL CATALOG:
    Destinations: Tokyo (Japan), Paris (France), Bali (Indonesia), New York City (USA)
    Flights: Economy Class, Business Class, First Class
    Accommodations: Standard Hotel, Boutique Hotel, Luxury Resort
    Activities: City Tour, Scuba Diving, Museum Pass
    """

@tool
def add_to_itinerary(item: str) -> str:
    """Add a travel service or item to the customer's itinerary."""
    global customer_itinerary
    customer_itinerary.append(item)
    return f"I've added '{item}' to your itinerary."

@tool
def clear_itinerary() -> str:
    """Clear all items from the customer's itinerary."""
    global customer_itinerary
    customer_itinerary.clear()
    return "Your itinerary has been cleared."

@tool
def get_itinerary() -> list[str]:
    """Get the current items in the customer's itinerary."""
    global customer_itinerary
    return customer_itinerary

@tool
def confirm_itinerary() -> str:
    """Confirm the itinerary with the customer."""
    global customer_itinerary
    if not customer_itinerary:
        return "Your itinerary is currently empty. Where would you like to go?"
    itinerary_string = ", ".join(customer_itinerary)
    return f"Your current itinerary contains: {itinerary_string}. Is this correct?"

@tool
def finalize_booking() -> str:
    """Place the final booking for the itinerary."""
    global customer_itinerary
    if not customer_itinerary:
        return "There's nothing in your itinerary to book."
    final_itinerary_summary = ", ".join(customer_itinerary)
    customer_itinerary.clear()
    return f"Your booking for '{final_itinerary_summary}' has been confirmed! Have a great trip."

# ==========================================
# 3. SYSTEM INSTRUCTION & AGENT SETUP
# ==========================================
system_instruction = """
You are a TravelBot, an interactive travel booking assistant.
A human will talk to you about the available travel destinations, flights, accommodations, and activities you have, and you will answer any questions about catalog items.
The customer will build an itinerary with 1 or more items from the travel catalog, which you will structure and send to the booking system after confirming the itinerary with the human.

Add items to the customer's itinerary with add_to_itinerary, and reset the itinerary with clear_itinerary.
To see the contents of the itinerary so far, call get_itinerary.
Always call confirm_itinerary with the user (double-check) before calling finalize_booking.
Once finalize_booking has returned, thank the user and wish them a great trip!
"""

travel_agent_tools = [get_travel_catalog, add_to_itinerary, clear_itinerary, get_itinerary, confirm_itinerary, finalize_booking]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
checkpointer = InMemorySaver()

travel_agent = create_agent(
    model=model,
    tools=travel_agent_tools,
    system_prompt=system_instruction,
    checkpointer=checkpointer
)

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
def chat_with_travel_agent():
    thread_id = "user_session_001" 
    config = {"configurable": {"thread_id": thread_id}}

    print("======================================================")
    print("🛫 Welcome to TravelBot! Type 'q' to quit.")
    print("======================================================\n")

    response = travel_agent.invoke(
        {"messages": [{"role": "user", "content": "Hello TravelBot! Introduce yourself."}]},
        config=config
    )
    print("TravelBot:", response["messages"][-1].content)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nTravelBot: Thank you for planning with us! Safe travels! ✈️")
            break

        response = travel_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )
        print("\nTravelBot:", response["messages"][-1].content)

# Run the program
if __name__ == "__main__":
    chat_with_travel_agent()

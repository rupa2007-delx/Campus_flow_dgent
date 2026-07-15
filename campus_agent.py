import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# =====================================================================
# TOOL 1: DRIVE RESOURCE RETRIEVER (For Notes & PYQs)
# =====================================================================
@tool
def fetch_drive_resources(resource_type: str) -> str:
    """
    Retrieves cloud drive links for academic resources.
    Use this when a student asks for 'PYQs', 'question papers', 'notes', or 'lectures'.
    - resource_type: Must be exactly 'PYQs' or 'notes'.
    """
    try:
        with open("campus_data.json", "r") as file:
            db = json.load(file)
        return db.get("learning_support", {}).get(resource_type, "Resource not found.")
    except Exception as e:
        return f"Error connecting to Drive layer: {str(e)}"


# =====================================================================
# TOOL 2: IRIS PORTAL SCRAPER (For Emails, Announcements, & Attendance)
# =====================================================================
@tool
def query_iris_portal(query_type: str) -> str:
    """
    Queries the IRIS portal data layer for administrative information.
    Use this when a student asks about attendance cards, dean emails, or notices.
    - query_type: Must be exactly 'emails', 'announcements', or 'attendance'.
    """
    try:
        with open("campus_data.json", "r") as file:
            db = json.load(file)
        
        if query_type in ['attendance', 'emails']:
            return db.get("announcements", {}).get(query_type, "No updates found.")
        else:
            return db.get("events_update", {}).get("TechFest", "No active announcements.")
    except Exception as e:
        return f"Error connecting to IRIS API: {str(e)}"


# =====================================================================
# TOOL 3: CAMPUS EVENT REGISTRATION SYSTEM
# =====================================================================
@tool
def execute_event_registration(event_name: str, student_name: str) -> str:
    """
    Registers a student directly for an upcoming technical or cultural event.
    Use this when a student explicitly asks to 'register', 'sign up', or 'join' an event.
    - event_name: The target festival (e.g., 'TechFest', 'CulturalNight').
    - student_name: The name of the student registering.
    """
    return f"✅ Success! {student_name} has been successfully registered for {event_name}."


# =====================================================================
# AGENT SETUP (Using lightweight Llama 3.2 for fast execution)
# =====================================================================
print("Loading Llama 3.2 Workspace Environment...")
llm = ChatOllama(model="llama3.2", temperature=0)

# Registering all 3 required tools to the LLM agent model 
campus_tools = [fetch_drive_resources, query_iris_portal, execute_event_registration]
agent_brain = llm.bind_tools(campus_tools)

# Prompt Engineering: Guiding Llama's reasoning paths
system_prompt = SystemMessage(content=(
    "You are CampusFlow AI, the intelligent assistant for the IRIS portal. "
    "Your job is to assist students by calling the appropriate tool at your disposal.\n"
    "1. If a student asks for study materials or PYQs, call 'fetch_drive_resources'. "
    "Set the resource_type argument to either 'PYQs' or 'notes'.\n"
    "2. If they ask about notices, emails, or attendance cards, call 'query_iris_portal'. "
    "Set the query_type argument to 'emails', 'announcements', or 'attendance'.\n"
    "3. If they want to sign up or register for an event, call 'execute_event_registration'. "
    "Identify the event_name and student_name.\n"
    "Only call one tool at a time. Be precise when extracting argument variables."
))

# Mapping tool names to actual functions for easy automated running
tool_dispatcher = {
    "fetch_drive_resources": fetch_drive_resources,
    "query_iris_portal": query_iris_portal,
    "execute_event_registration": execute_event_registration
}

# =====================================================================
# RUNNING ALL THREE TEST CASES BACK-TO-BACK
# =====================================================================
test_queries = [
    "Can you register Rupas for TechFest?",
    "Where can I find the Machine Learning notes?",
    "Are there any updates on attendance cards?"
]

for i, query in enumerate(test_queries, 1):
    print("\n" + "="*70)
    print(f"TEST CASE {i}: Processing student request: '{query}'")
    print("="*70)
    
    # Send the student's request to the agent
    ai_analysis = agent_brain.invoke([system_prompt, HumanMessage(content=query)])

    if ai_analysis.tool_calls:
        for call in ai_analysis.tool_calls:
            tool_name = call['name']
            tool_args = call['args']
            print(f"🎯 Llama 3.2 matched the query to: {tool_name}")
            print(f"⚙️ Extracted Parameters: {tool_args}")
            
            # Dynamically execute the correct tool function
            if tool_name in tool_dispatcher:
                output = tool_dispatcher[tool_name].invoke(tool_args)
                print(f"🤖 [Database Output]: {output}")
            else:
                print(f"❌ Error: Tool '{tool_name}' is not registered in the dispatcher.")
    else:
        print(f"🤖 [Direct AI Response]: {ai_analysis.content}")

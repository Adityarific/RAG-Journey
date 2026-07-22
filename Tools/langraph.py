import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# 1. Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. Initialize the Google Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Create the agent using LangGraph's wrapper
agent = create_react_agent(
    model=model,
    tools=[],  # Empty list of tools as shown in your image
    state_modifier="You are a helpful assistant."
)

# 4. Display the agent object
agent

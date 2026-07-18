import os
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from langchain_google_community.gmail.toolkit import GmailToolkit
from langchain_google_community.gmail.utils import build_resource_service
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 1. Load variables from your .env file
load_dotenv()

SCOPES = ["https://googleapis.com"]

# 2. Build the client configuration using your specific env keys
client_config = {
    "installed": {
        "client_id": os.getenv("CLIENT_ID"),
        "client_secret": os.getenv("CLIENT_SECRET"),
        "auth_uri": "https://google.com",
        "token_uri": "https://googleapis.com",
        "redirect_uris": ["http://localhost"]
    }
}

# 3. Handle Gmail token generation and user login
creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # Uses your .env config structure dynamically without a credentials.json file
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(port=0)
    
    # Saves token.json so login is only required once
    with open("token.json", "w") as token:
        token.write(creds.to_json())

# 4. Bind credentials to LangChain Gmail Toolkit
api_resource = build_resource_service(credentials=creds)
toolkit = GmailToolkit(api_resource=api_resource)
tools = toolkit.get_tools()

# 5. Connect the Novita AI Endpoint using ChatOpenAI
# It passes your Hugging Face Token as the API key to authenticate with Novita
agent_llm = ChatOpenAI(
    model="google/gemma-4-31B-it",  
    openai_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
    openai_api_base="https://novita.ai",
    temperature=0.1
)

# 6. Build the ReAct Agent
agent_executor = create_react_agent(agent_llm, tools)

# 7. Test Query
query = "Check my latest unread emails."
events = agent_executor.stream({"messages": [("user", query)]}, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()

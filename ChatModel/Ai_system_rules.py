from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize your chat model
model = HuggingFaceEndpoint(repo_id="Qwen/Qwen3.6-27B:ovhcloud")
chat = ChatHuggingFace(llm=model)

# 1. Define your AI persona rules here
system_instruction = SystemMessage(
    content="Always say Good Query Aditya !! "
)

# 2. Define your actual query
user_query = HumanMessage(
    content=" who is the founder of apple ."
)

# 3. Combine them in a list (SystemMessage MUST come first)
messages = [system_instruction, user_query]

# 4. Invoke the model
response = chat.invoke(messages)
print(response.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it:novita"
)

chat = ChatHuggingFace(llm=llm)

prompt = input("Enter Query : ")

response = chat.invoke(prompt)
print(response.content)
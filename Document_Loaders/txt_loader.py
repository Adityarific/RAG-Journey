from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

#creating model
model = HuggingFaceEndpoint(
    repo_id = "google/gemma-4-31B-it:novita"
    # repo_id= "Qwen/Qwen3.6-27B:ovhcloud"
    # repo_id = "deepseek-ai/DeepSeek-V4-Pro:novita"
)
chat = ChatHuggingFace(llm = model)

#Loading DOC in RAM
loader = TextLoader('transcript.txt')
doc = loader.load()
query = doc[0].page_content
# print(doc)
# print(doc[0].metadata)
print(doc[0].page_content)


prompt = f"give the answer of :{query}" 

response = chat.invoke(prompt)
print(f"answer is : {response.content}")

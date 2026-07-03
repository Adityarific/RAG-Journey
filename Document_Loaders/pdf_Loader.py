from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

model= HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it:novita"
)

chat = ChatHuggingFace( llm  = model )

# loaded PDF in system
loader = PyPDFLoader('docs/Steve_Jobs.pdf')
doc=loader.load()
query = doc[2].page_content 
print(doc[2].page_content)


prompt = f"summarise this phara in 5 points {query}"

response = chat.invoke(prompt)

print(f"summarized version is {response.content}")
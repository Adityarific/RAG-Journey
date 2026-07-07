from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.retrievers import WikipediaRetriever

from dotenv import load_dotenv

load_dotenv()

# Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en",
    load_all_available_meta=False,
)

query = input("Enter Query : ")

# Fetch documents
docs = retriever.invoke(query)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it:novita"
)

chat = ChatHuggingFace(llm=llm)

prompt = docs.page_content[:500]

response = chat.invoke(prompt)
print(response.content)
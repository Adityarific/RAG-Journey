from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import  InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()

# Sample dataset formatted for LangChain RAG pipelines
from langchain_core.documents import Document

docs = [
    Document(
        page_content="The Sun is a yellow dwarf star at the center of the Solar System,",
        metadata={"source": "SOLAR-001", "body_type": "Star", "name": "The Sun"}
    ),
    Document(
        page_content="Mercury is the smallest planet in the Solar System and the closest to the Sun. ranging from -180 degrees Celsius at night to 430 degrees " \
        "Celsius during the day. Its surface is heavily cratered, resembling Earth's Moon,.",
        metadata={"source": "SOLAR-002", "body_type": "Terrestrial Planet", "name": "Mercury"}
    ),
    Document(
        page_content="Venus is often called Earth's twin due to its similar size and mass, but it features a runaway greenhouse effect. " \
        "Its thick atmosphere is carbon dioxide, trapping solar heat and creating surface temperatures of 465 degrees Celsius, making" \
        " it the hottest planet in the Solar System. ",
        metadata={"source": "SOLAR-003", "body_type": "Terrestrial Planet", "name": "Venus"}
    )
]

# 1. Initialize the embedding model with correct instruction prefixes
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    encode_kwargs={"prompt": "passage: "},        # Prefixed to documents during indexing
    query_encode_kwargs={"prompt": "query: "}    # Prefixed to user input during retrieval
)

print("embeddings done")

# 2. FIXED: Initialize Chroma by passing the docs list directly using the proper method
vector_store =  InMemoryVectorStore.from_documents(
    documents=docs, 
    embedding=embeddings)

print("vectors done")

# 3. FIXED: Adjusted argument from `text=` to the correct plural parameter `texts=`
vector_store.add_texts(texts=["LangGraph orchestrates cyclical AI workflows seamlessly."])

results = vector_store.similarity_search(
    query= "smallest planet in solar system",
    k = 2,
    
)
print(results[0].metadata['name'])


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# 1. Load and Split Document
loader = PyPDFLoader('attention.pdf')
context = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunk = splitter.split_documents(context)

# 2. Setup Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"prompt": "passage"},
    query_encode_kwargs={"prompt": "query"}
)

vector_store = FAISS.from_documents(chunk, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# 3. Initialize LLM and Chat Model
llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it:novita")
chat = ChatHuggingFace(llm=llm)

# 4. Define Prompt (Cleaned String format)
system_prompt = (
    "Always say Good Query Aditya !! "
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Helper function to format retrieved documents into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Construct the LCEL RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,  # Fetch documents and convert to clean text
        "input": RunnablePassthrough()       # Forward user string text directly
    }
    | prompt                                 # Format into system/human schema
    | chat                                   # Call HuggingFace Chat Engine
    | StrOutputParser()                      # Convert the LLM Message block back to text string
)

# 6. Execute User Query
user_query = input('Enter query: ')
response_text = rag_chain.invoke(user_query)

print("\nAnswer:")
print(response_text)

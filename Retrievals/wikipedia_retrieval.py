# Wikipedia Retriever
# A Wikipedia Retriever is a retriever that queries the Wikipedia API to fetch relevant content for a given query.
from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en",
    load_all_available_meta=False,
)

query = "who is the founder of apple?"

# Fetch documents
docs = retriever.invoke(query)

# Clean, structured print loop
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(f"Source Title: {doc.metadata.get('title', 'Unknown')}")
    print(f"Snippet:\n{doc.page_content[:500]}...")  # Truncated for readability

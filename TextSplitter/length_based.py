from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('Steve_Jobs.pdf')
docs = loader.load()
query = docs[2].page_content

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
    )
chunks = splitter.split_text(query)
print(chunks)

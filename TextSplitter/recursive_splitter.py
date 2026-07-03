# Recursive splitter split the text on the basis on hirarchical structure of the text.
# It will first try to split the text into 
# paragraphs, 
# then sentences, 
# and finally words.
# This is useful for splitting long documents into smaller chunks while preserving the context of the text.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Steve_Jobs.pdf')
docs = loader.load()
text = docs[2].page_content

print('text loaded')

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10,
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk)
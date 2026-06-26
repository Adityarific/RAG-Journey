from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Steve_Jobs.pdf')

doc=loader.load()
print(doc[2].page_content)
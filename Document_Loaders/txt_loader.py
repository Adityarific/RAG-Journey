from langchain_community.document_loaders import TextLoader
loader = TextLoader('test.txt')
doc = loader.load()
print(doc)
print(doc[0].metadata)
print(doc[0].page_content)

from langchain_community.tools import ShellTool

shell = ShellTool()
result = shell.invoke({"commands": ["code ."]})
print(result)

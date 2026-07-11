from langchain_community.tools import ShellTool

# Initialize the tool
shell = ShellTool()

# Correct way to invoke by passing arguments as a structured dictionary
result = shell.invoke({"commands": ["whoami"]})

# Print the system user output
print(result)
